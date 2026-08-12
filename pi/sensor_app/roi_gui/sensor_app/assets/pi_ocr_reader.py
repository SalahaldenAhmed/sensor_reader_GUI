# pi_ocr_reader.py
# ONNX recognition-only OCR (PP-OCRv5 mobile rec, converted via paddle2onnx).
# Runs on onnxruntime — no PaddlePaddle. YOLO upstream crops the display.
import cv2, re
import numpy as np
import onnxruntime as ort

class OCRReader:
    def __init__(self, model_path="en_PP-OCRv5_mobile_rec.onnx",
                       dict_path="ppocrv5_dict.txt", rec_h=48):
        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.rec_h = rec_h
        # CTC label layout: [blank] + dict chars + [space]
        with open(dict_path, encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        self.labels = ["blank"] + chars + [" "]   # index 0..437

    def _preprocess(self, crop_bgr, target_w=320):
        # PP-OCRv5 rec: resize to height 48, keep aspect, pad width.
        h, w = crop_bgr.shape[:2]
        ratio = w / float(h)
        new_w = int(self.rec_h * ratio)
        new_w = min(new_w, target_w)
        img = cv2.resize(crop_bgr, (new_w, self.rec_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5            # normalize to [-1, 1]
        img = img.transpose(2, 0, 1)               # HWC -> CHW
        padded = np.zeros((3, self.rec_h, target_w), np.float32)
        padded[:, :, :new_w] = img
        return padded[None]                        # (1,3,48,target_w)

    def _ctc_decode(self, logits):
        # logits: (T, num_classes). Greedy: argmax per step, collapse repeats, drop blank.
        idx = logits.argmax(axis=1)
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0:               # 0 = blank
                out.append(self.labels[i])
            prev = i
        return "".join(out)

    def read(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        inp = self._preprocess(crop_bgr)
        out = self.sess.run(None, {self.inp_name: inp})[0]
        text = self._ctc_decode(out[0])            # drop batch dim
        return text

    @staticmethod
    def clean_number(raw, kind="temp"):
        if not raw:
            return None
        negative = "-" in raw
        digits = re.sub(r"[^\d]", "", raw)
        if not digits:
            return None
        if kind == "temp":
            digits = digits[:3]
            value = f"{digits[:-1]}.{digits[-1]}" if len(digits) > 1 else digits
            return f"-{value}" if negative else value
        if kind == "humidity":
            return digits[:2]
        if kind == "time":
            digits = digits[:4]
            if len(digits) == 3:
                return f"{digits[0]}:{digits[1:]}"
            if len(digits) == 4:
                return f"{digits[:2]}:{digits[2:]}"
            return digits
        return digits


CLASS_KIND = {"Time": "time", "Humidity": "humidity",
              "IN_TEMP": "temp", "OUT_TEMP": "temp"}

if __name__ == "__main__":
    import sys
    reader = OCRReader()
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("usage: python3 pi_ocr_reader.py <crop.jpg>")
    else:
        print(f"raw OCR: {reader.read(img)!r}")
