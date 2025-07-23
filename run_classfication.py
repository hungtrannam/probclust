import numpy as np
import matplotlib.pyplot as plt
import os

from utils.integral import grid
from data.data_loader import generateGauss
from Models.classification import Logistics, SVM
from utils.kernel import kernel
from utils.vis import *

def generate_gaussian_basis(x_grid, M, sigma=None):
    centers = np.linspace(x_grid.min(), x_grid.max(), M)
    sigma = sigma or (x_grid.max() - x_grid.min()) / (2 * M)
    return [np.exp(-0.5 * ((x_grid - c) / sigma) ** 2) for c in centers]

def prepare_logistic_features(pdfs, basis_functions, x_grid):
    moments = np.array([
        [np.trapezoid(f * psi, x_grid) for psi in basis_functions]
        for f in pdfs
    ])
    X_design = np.hstack([np.ones((moments.shape[0], 1)), moments])
    return X_design

def predict_new_sample_logistic(new_pdf, log_model, basis_functions, x_grid):
    moment = np.array([np.trapezoid(new_pdf * psi, x_grid) for psi in basis_functions])
    X_new = np.hstack([1.0, moment])
    prob = log_model.predict_proba(X_new.reshape(1, -1))[0]
    pred = log_model.predict(X_new.reshape(1, -1))[0]
    return pred, prob

def predict_new_sample_svm(new_pdf, all_pdfs, svm_model, kc):
    K_full = kc.L1(np.vstack([all_pdfs, new_pdf[None, :]]))
    K_new = K_full[-1, :-1].reshape(1, -1)
    prob = svm_model.predict_proba(K_new)[0] # probability for class 1
    pred = svm_model.predict(K_new)[0]
    return pred, prob

import shap

def explain_logistic_model(log_model, X_design, basis_functions, x_grid):
    explainer = shap.Explainer(log_model, X_design)
    shap_values = explainer(X_design)

    # Vẽ biểu đồ summary plot cho toàn bộ tập
    shap.summary_plot(shap_values, X_design, show=False)
    plt.savefig("figs/shap_summary_logistic.png", bbox_inches='tight')

    # SHAP values cho sample mới
    new_pdf = generateGauss([3], [0.9], x_grid).ravel()
    moment = np.array([np.trapezoid(new_pdf * psi, x_grid) for psi in basis_functions])
    X_new = np.hstack([1.0, moment])
    shap_value_new = explainer(X_new.reshape(1, -1))

    # Vẽ force plot cho sample mới
    shap.plots.force(shap_value_new[0], matplotlib=True, show=False)
    plt.savefig("figs/shap_force_logistic.pdf", bbox_inches='tight')



def run_and_plot(x_grid, class1_data, class2_data, all_pdfs, y, new_pdf):
    n_A, n_B = len(class1_data), len(class2_data)

    # Logistic Regression
    print("\n=== Logistic Regression ===")
    M = 10
    basis_functions = generate_gaussian_basis(x_grid, M)
    X_design = prepare_logistic_features(all_pdfs, basis_functions, x_grid)
    log_model = Logistics.Model(n_iter=1000, l1_penalty=0.1, verbose=False)
    log_model.fit(X_design, y)
    decision_values = log_model.predict_proba(X_design)
    preds = log_model.predict(X_design)
    print(preds)
    acc = np.mean(preds == y)
    print(f"[LOGISTIC] Accuracy: {acc*100:.2f}%")

    plot_log_function(x_grid, sum(log_model.coef_[j] * basis_functions[j] for j in range(M)), "figs/beta_log.pdf")
    plot_decision(x_grid, all_pdfs, decision_values, n_A, n_B, 'figs/prob_log.pdf')
    explain_logistic_model(log_model, X_design, basis_functions, x_grid)


    pred_log, prob_log = predict_new_sample_logistic(new_pdf, log_model, basis_functions, x_grid)
    print(f"[LOGISTIC] New sample → Class: {pred_log}, Probability (class 1): {prob_log:.4f}")

    # === SVM ===
    print("\n=== SVM ===")
    kc = kernel(h=0.01, Dim=1)
    K_train = kc.L1(all_pdfs)  # (n_train, n_train)

    svm_model = SVM.Model(C=1) 
    svm_model.fit(K_train, y)

    # Predict trên training set
    preds_svm = svm_model.predict(K_train)
    acc_svm = np.mean(preds_svm == y)
    print(f"[SVM] Accuracy: {acc_svm * 100:.2f}%")

    # Lấy probability (class 1) trên training set
    probas_svm = svm_model.predict(K_train)
    print(probas_svm)
    plot_decision(x_grid, all_pdfs, probas_svm, n_A, n_B, 'figs/prob_svm.pdf')

    # Predict sample mới
    # Tạo kernel giữa (train, new_sample) → (n_train,)
    K_full = kc.L1(np.vstack([all_pdfs, new_pdf[None, :]]))  # (n_train+1, n_train+1)
    K_new = K_full[:-1, -1][:, None]  # (n_train, 1)

    # Predict class và probability
    pred_svm = svm_model.predict(K_new)[0]
    print(f"[SVM] New sample → Class: {pred_svm}")
    


if __name__ == "__main__":
    os.makedirs('figs', exist_ok=True)
    h = 0.01
    x_grid, _ = grid(h, start=-2, end=12)

    mu_A = [1, 1.5, 2, 2.5, 6]
    sig_A = [0.4] * 5
    mu_B = [4.5, 7.5, 8.5, 8, 8.5]
    sig_B = [0.9] * 5

    class1_data = generateGauss(mu_A, sig_A, x_grid)
    class2_data = generateGauss(mu_B, sig_B, x_grid)
    all_pdfs = np.vstack([class1_data, class2_data])
    y = np.array([0]*len(class1_data) + [1]*len(class2_data))

    # New sample
    new_pdf = generateGauss([3], [0.9], x_grid).ravel()

    run_and_plot(x_grid, class1_data, class2_data, all_pdfs, y, new_pdf)
    
