# hybrid_pipeline.py
import numpy as np

class HybridFraudPipeline:
    def __init__(self, scaler, dt, cb, xgb, kmeans, iso, hybrid_cb):
        self.scaler = scaler
        self.dt = dt
        self.cb = cb
        self.xgb = xgb
        self.kmeans = kmeans
        self.iso = iso
        self.hybrid_cb = hybrid_cb

    def preprocess(self, X):
        if 'isFraud' in X.columns:
            X = X.drop(columns=['isFraud'])
        X_scaled = self.scaler.transform(X)
        dt_pred = self.dt.predict(X_scaled).reshape(-1, 1)
        cb_pred = self.cb.predict(X_scaled).reshape(-1, 1)
        xgb_pred = self.xgb.predict(X_scaled).reshape(-1, 1)
        clusters = self.kmeans.predict(X_scaled).reshape(-1, 1)
        anomaly = self.iso.decision_function(X_scaled).reshape(-1, 1)
        X_hybrid = np.hstack((X_scaled, dt_pred, cb_pred, xgb_pred, clusters, anomaly))
        return X_hybrid

    def predict(self, X):
        X_hybrid = self.preprocess(X)
        return self.hybrid_cb.predict(X_hybrid)
