import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os

class Model:
    """
    Naive Bayes cho dữ liệu hàm mật độ xác suất (PDF).
    """

    def __init__(self, x_grid):
        self.x_grid = x_grid
        self.classes_ = None
        self.class_priors_ = {}
        self.class_pdfs_ = {}      # Các PDF của từng lớp
        self.class_mean_pdf_ = {}  # PDF trung bình cho mỗi lớp

    def fit(self, pdfs, labels, priors=None):
        """
        Huấn luyện Naive Bayes cho PDF.
        """
        pdfs = np.asarray(pdfs)
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)

        for c in self.classes_:
            class_pdfs = pdfs[labels == c]
            self.class_pdfs_[c] = class_pdfs
            self.class_mean_pdf_[c] = np.mean(class_pdfs, axis=0)

        # Thiết lập prior mặc định (nếu không truyền priors)
        if priors is None:
            self.class_priors_ = {
                c: self.class_pdfs_[c].shape[0] / pdfs.shape[0]
                for c in self.classes_
            }
        else:
            self.set_priors(priors)
        return self

    def set_priors(self, priors):
        """
        Thiết lập custom priors, priors: dict {class: prior_value}
        """
        for c in self.classes_:
            if c in priors:
                self.class_priors_[c] = priors[c]
            else:
                # Nếu không có trong priors, giữ nguyên hoặc phân đều
                self.class_priors_[c] = 1.0 / len(self.classes_)
        return self

    def _f_h_given_class(self, h, class_label):
        pdfs_class = self.class_pdfs_[class_label]
        values = [trapezoid(f_j * h, self.x_grid) for f_j in pdfs_class]
        return np.mean(values)

    def predict_proba(self, pdfs, priors=None):
        """
        Dự đoán xác suất P(c_k | h) cho mỗi PDF.
        Có thể truyền priors custom (dict {class: prior}).
        """
        pdfs = np.asarray(pdfs)
        proba = np.zeros((pdfs.shape[0], len(self.classes_)))

        # Sử dụng prior tùy chỉnh nếu truyền vào
        class_priors = self.class_priors_.copy()
        if priors is not None:
            for c in self.classes_:
                class_priors[c] = priors.get(c, class_priors[c])

        for i, h in enumerate(pdfs):
            scores = np.array([
                class_priors[c] * self._f_h_given_class(h, c)
                for c in self.classes_
            ])
            proba[i] = scores / scores.sum()
        return proba

    def predict(self, pdfs, priors=None):
        proba = self.predict_proba(pdfs, priors=priors)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def explain(self, h, priors=None, savefile=False):
        """
        Giải thích quyết định phân loại cho một PDF h.
        Hiển thị cả likelihood và posterior (có tính đến prior).
        """
        h = np.asarray(h).ravel()  # Đảm bảo h có dạng 1D

        # Sử dụng prior mới nếu cung cấp
        class_priors = self.class_priors_.copy()
        if priors is not None:
            for c in self.classes_:
                class_priors[c] = priors.get(c, class_priors[c])

        # Tính likelihood và posterior
        scores = {}
        likelihoods = {}
        for c in self.classes_:
            likelihoods[c] = self._f_h_given_class(h, c)
            scores[c] = class_priors[c] * likelihoods[c]
        total = sum(scores.values())
        posterior = {c: scores[c] / total for c in self.classes_}

        if savefile:
            from utils.vis import temp
            plt.figure(figsize=(6, 5))
            temp(20)

            # Vẽ PDF cần phân loại
            plt.plot(
                self.x_grid, h, label=r"$h(x)$",
                color="black", linewidth=3, linestyle="-"
            )

            # Kiểu màu cho từng class
            gray_styles = [("black", ":"), ("gray", "--")]

            # Vẽ PDF trung bình cho từng lớp
            for idx, c in enumerate(self.classes_):
                color, style = gray_styles[idx % len(gray_styles)]
                label_text = f"Class {c} (prior = {class_priors[c]:.2f})"
                plt.plot(
                    self.x_grid,
                    self.class_mean_pdf_[c],
                    label=label_text,
                    color=color,
                    linestyle=style,
                    linewidth=2
                )
                plt.fill_between(
                    self.x_grid, self.class_mean_pdf_[c],
                    alpha=0.2, color=color
                )

            plt.legend(fontsize=13)
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
            plt.savefig(savefile, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {savefile}")
            plt.close()


        return {"likelihood": likelihoods, "posterior": posterior}

