from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

import cv2
import numpy as np
import torch


@dataclass
class SimilarityResult:
    similarity: float
    confidence: float
    matching_points: int
    debug: dict = field(default_factory=dict)


class BaseSimilarityEstimator(ABC):
    @abstractmethod
    def compute_similarity(self, reference_frame: torch.Tensor, middle_frame: torch.Tensor, return_debug: bool = False) -> SimilarityResult:
        pass


def chw_to_gray_u8(chw: torch.Tensor) -> np.ndarray:
    """
    chw: [C, H, W] tensor, value range can be [-1, 1] or [0, 1]
    Returns: HxW uint8 grayscale image (0~255)
    """
    x = chw.detach().cpu().float()

    if x.min() < 0:
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:
        x = (x * 255.0).clamp(0, 255.0)

    if x.shape[0] == 3:
        r, g, b = x[0], x[1], x[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = x[0]

    return gray.byte().numpy()


class ORBSimilarityEstimator(BaseSimilarityEstimator):
    def __init__(self, ratio_thresh: float = 0.8, min_good: int = 30):
        self.ratio_thresh = ratio_thresh
        self.min_good = min_good

    def compute_similarity(self, reference_frame: torch.Tensor, middle_frame: torch.Tensor, return_debug: bool = False) -> SimilarityResult:
        """
        ORB + RANSAC similarity. This preserves the original StableWorld ORB
        parameters and scoring logic exactly.
        """
        start_time = time.perf_counter()
        img1 = chw_to_gray_u8(reference_frame)
        img2 = chw_to_gray_u8(middle_frame)

        def finish(score: float, debug: dict | None = None) -> SimilarityResult:
            debug = debug or {}
            debug.setdefault("matching_points", 0)
            debug.setdefault("inliers_h", 0)
            debug.setdefault("inliers_f", 0)
            debug.setdefault("ratio_h", 0.0)
            debug.setdefault("ratio_f", 0.0)
            debug.setdefault("match_image", None)
            debug["orb_runtime_ms"] = (time.perf_counter() - start_time) * 1000.0
            if not return_debug:
                debug["match_image"] = None
            return SimilarityResult(
                similarity=float(score),
                confidence=float(score),
                matching_points=int(debug.get("matching_points", 0)),
                debug=debug,
            )

        orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        if d1 is None or d2 is None:
            return finish(0.0)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)

        good = []
        for matches in knn:
            if len(matches) < 2:
                continue
            m, n = matches
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        if len(good) < 7:
            match_image = None
            if return_debug:
                match_image = cv2.drawMatches(img1, k1, img2, k2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return finish(0.0, {"matching_points": len(good), "match_image": match_image})

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

        inH = int(maskH.sum()) if maskH is not None else 0
        inF = int(maskF.sum()) if isinstance(maskF, np.ndarray) and maskF.size > 0 else 0

        denom = max(len(good), 1)
        ratioH = inH / denom
        ratioF = inF / denom

        if len(good) < self.min_good:
            scale = len(good) / float(self.min_good)
            score = float(max(ratioH, ratioF) * scale)
        else:
            score = float(max(ratioH, ratioF))

        match_image = None
        if return_debug:
            inlier_mask = maskH if ratioH >= ratioF else maskF
            if isinstance(inlier_mask, np.ndarray) and inlier_mask.size > 0:
                matches_mask = inlier_mask.ravel().astype(int).tolist()
            else:
                matches_mask = None
            match_image = cv2.drawMatches(
                img1,
                k1,
                img2,
                k2,
                good,
                None,
                matchesMask=matches_mask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

        return finish(score, {
            "matching_points": len(good),
            "inliers_h": inH,
            "inliers_f": inF,
            "ratio_h": float(ratioH),
            "ratio_f": float(ratioF),
            "match_image": match_image,
        })


def orb_ransac_score_chw(chwA: torch.Tensor, chwB: torch.Tensor, ratio_thresh: float = 0.8, min_good: int = 30, return_debug: bool = False):
    result = ORBSimilarityEstimator(ratio_thresh=ratio_thresh, min_good=min_good).compute_similarity(
        chwA,
        chwB,
        return_debug=return_debug,
    )
    if not return_debug:
        return result.similarity
    return result.similarity, result.debug


def chw_to_rgb_float_tensor(chw: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    x = chw.detach().float()
    if x.min() < 0:
        x = ((x + 1.0) * 0.5).clamp(0, 1)
    else:
        x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.to(device)


def chw_to_rgb_u8(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().cpu().float()
    if x.min() < 0:
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:
        x = (x * 255.0).clamp(0, 255.0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.byte().permute(1, 2, 0).numpy()


class LightGlueSimilarityEstimator(BaseSimilarityEstimator):
    def __init__(
        self,
        features: str = "superpoint",
        max_num_keypoints: int = 2048,
        spatial_alpha: float = 0.0,
        device: str | None = None,
    ):
        self.features = features
        self.max_num_keypoints = max_num_keypoints
        self.spatial_alpha = float(spatial_alpha)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = None
        self.matcher = None

    def _lazy_init(self):
        if self.extractor is not None and self.matcher is not None:
            return
        try:
            from lightglue import LightGlue, SuperPoint, DISK, SIFT
        except ImportError as exc:
            raise ImportError(
                "LightGlueSimilarityEstimator requires the lightglue package. "
                "Install it before using --similarity_estimator lightglue."
            ) from exc

        if self.features == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        elif self.features == "disk":
            self.extractor = DISK(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        elif self.features == "sift":
            self.extractor = SIFT(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        else:
            raise ValueError(f"Unsupported LightGlue feature extractor: {self.features}")
        self.matcher = LightGlue(features=self.features).eval().to(self.device)

    def compute_similarity(self, reference_frame: torch.Tensor, middle_frame: torch.Tensor, return_debug: bool = False) -> SimilarityResult:
        """
        Semantic matching similarity based on LightGlue correspondences.
        MemoryScheduler still receives the same SimilarityResult interface.
        """
        start_time = time.perf_counter()
        self._lazy_init()

        image0 = chw_to_rgb_float_tensor(reference_frame, self.device)[None]
        image1 = chw_to_rgb_float_tensor(middle_frame, self.device)[None]

        with torch.no_grad():
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

        keypoints0 = feats0["keypoints"][0].detach().cpu().float()
        keypoints1 = feats1["keypoints"][0].detach().cpu().float()
        matches = matches01["matches"][0].detach().cpu().long()
        scores = matches01.get("scores", None)
        if scores is not None:
            scores = scores[0].detach().cpu().float()
        else:
            scores = torch.ones(matches.shape[0], dtype=torch.float32)

        matching_points = int(matches.shape[0])
        if matching_points == 0:
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return SimilarityResult(
                similarity=0.0,
                confidence=0.0,
                matching_points=0,
                debug={
                    "lightglue_runtime_ms": runtime_ms,
                    "match_image": None,
                    "matching_heatmap": None,
                    "confidence_distribution": None,
                    "confidence_scores": [],
                    "displacement_field": None,
                    "motion_vector": None,
                    "semantic_similarity": 0.0,
                    "final_similarity": 0.0,
                    "spatial_alpha": self.spatial_alpha,
                    "average_displacement": 0.0,
                    "median_displacement": 0.0,
                    "maximum_displacement": 0.0,
                },
            )

        matched_keypoints0 = keypoints0[matches[:, 0]]
        matched_keypoints1 = keypoints1[matches[:, 1]]
        displacement = matched_keypoints1 - matched_keypoints0
        displacement_norm = torch.linalg.norm(displacement, dim=1)

        confidence = float(scores.mean().item())
        semantic_similarity = confidence
        average_displacement = float(displacement_norm.mean().item())
        median_displacement = float(displacement_norm.median().item())
        maximum_displacement = float(displacement_norm.max().item())
        final_similarity = float(max(0.0, semantic_similarity - self.spatial_alpha * average_displacement))
        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        debug = {
            "lightglue_runtime_ms": runtime_ms,
            "matching_points": matching_points,
            "confidence_scores": scores.numpy().astype(float).tolist(),
            "feature_backend": self.features,
            "semantic_similarity": semantic_similarity,
            "final_similarity": final_similarity,
            "spatial_alpha": self.spatial_alpha,
            "average_displacement": average_displacement,
            "median_displacement": median_displacement,
            "maximum_displacement": maximum_displacement,
            "match_image": None,
            "matching_heatmap": None,
            "confidence_distribution": None,
            "displacement_field": None,
            "motion_vector": None,
        }
        if return_debug:
            debug.update(self._build_visualizations(reference_frame, middle_frame, keypoints0, keypoints1, matches, scores))

        return SimilarityResult(
            similarity=float(final_similarity),
            confidence=float(confidence),
            matching_points=matching_points,
            debug=debug,
        )

    def _build_visualizations(self, reference_frame, middle_frame, keypoints0, keypoints1, matches, scores):
        ref = chw_to_rgb_u8(reference_frame)
        cur = chw_to_rgb_u8(middle_frame)
        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
        cur_bgr = cv2.cvtColor(cur, cv2.COLOR_RGB2BGR)
        h0, w0 = ref_bgr.shape[:2]
        h1, w1 = cur_bgr.shape[:2]
        canvas_h = max(h0, h1)
        match_image = np.zeros((canvas_h, w0 + w1, 3), dtype=np.uint8)
        match_image[:h0, :w0] = ref_bgr
        match_image[:h1, w0:w0 + w1] = cur_bgr

        heatmap = np.zeros((h1, w1), dtype=np.float32)
        displacement_field = cur_bgr.copy()
        motion_vector = np.zeros_like(cur_bgr) + 255
        score_np = scores.numpy()
        if len(score_np) > 0:
            denom = max(float(score_np.max() - score_np.min()), 1e-6)
            norm_scores = (score_np - float(score_np.min())) / denom
        else:
            norm_scores = score_np

        for idx, match in enumerate(matches.numpy()):
            p0 = keypoints0[int(match[0])].numpy()
            p1 = keypoints1[int(match[1])].numpy()
            conf = float(norm_scores[idx]) if len(norm_scores) > idx else 1.0
            color = (0, int(255 * conf), int(255 * (1.0 - conf)))
            pt0 = (int(round(p0[0])), int(round(p0[1])))
            pt1 = (int(round(p1[0])) + w0, int(round(p1[1])))
            cv2.line(match_image, pt0, pt1, color, 1, cv2.LINE_AA)
            cv2.circle(match_image, pt0, 2, color, -1)
            cv2.circle(match_image, pt1, 2, color, -1)
            x1 = int(np.clip(round(p1[0]), 0, w1 - 1))
            y1 = int(np.clip(round(p1[1]), 0, h1 - 1))
            heatmap[y1, x1] += float(scores[idx].item())
            start_pt = (int(round(p0[0])), int(round(p0[1])))
            end_pt = (int(round(p1[0])), int(round(p1[1])))
            cv2.arrowedLine(displacement_field, start_pt, end_pt, color, 1, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(displacement_field, end_pt, 2, color, -1)
            cv2.arrowedLine(motion_vector, start_pt, end_pt, color, 1, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(motion_vector, start_pt, 2, (80, 80, 80), -1)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (0, 0), 9)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        matching_heatmap = cv2.addWeighted(cur_bgr, 0.55, heatmap_color, 0.45, 0)

        hist = np.zeros((320, 480, 3), dtype=np.uint8) + 255
        bins = np.linspace(0, 1, 21)
        counts, _ = np.histogram(np.clip(score_np, 0, 1), bins=bins)
        max_count = max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            x0 = 30 + i * 21
            x1 = x0 + 16
            y1 = 280
            y0 = y1 - int((count / max_count) * 230)
            cv2.rectangle(hist, (x0, y0), (x1, y1), (50, 120, 220), -1)
        cv2.putText(hist, "LightGlue confidence distribution", (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(hist, "0", (28, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(hist, "1", (438, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        return {
            "match_image": match_image,
            "matching_heatmap": matching_heatmap,
            "confidence_distribution": hist,
            "displacement_field": displacement_field,
            "motion_vector": motion_vector,
        }


def build_similarity_estimator(name: str = "orb", lightglue_spatial_alpha: float = 0.0) -> BaseSimilarityEstimator:
    normalized = (name or "orb").lower()
    if normalized == "orb":
        return ORBSimilarityEstimator()
    if normalized == "lightglue":
        return LightGlueSimilarityEstimator(spatial_alpha=lightglue_spatial_alpha)
    raise ValueError(f"Unknown similarity estimator: {name}")
