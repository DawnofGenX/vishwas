# _urlphish_vendor — vendored classical-ML URL-phishing classifier

Source: `malindard/phishing-checker-flask` (MIT) — https://github.com/malindard/phishing-checker-flask
Archived: 2026-08-26 (cloned depth-1; repo pushed 2025-08-26).

Files vendored:
- url_phishing_model.pkl  (2,620,498 B) — XGBoost classifier (joblib)
- scaler.pkl              (1,743 B)     — StandardScaler fit on training features
- selected_features.pkl   (357 B)       — 23 feature names in model order
- model_info.pkl          (553 B)       — model metadata (model_type=XGBoost, params)
- url_feature_extractor.py(25,641 B)    — 40+ lexical/network feature fns

Prediction contract (api/api_url.py url_predict_phishing):
    X = pd.DataFrame([features])[selected_features]  # 23 cols, order fixed
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]        # index 1 = PHISHING prob
High = phishing probability (positive / risky), matching fusion's positive-weight
convention.

License: MIT (upstream). Integration posture:
- 16 lexical features deterministic/offline; 7 network-scrape features
  (nb_hyperlinks, ratio_intHyperlinks, empty_title, domain_in_title,
  domain_age, google_index, page_rank) are gated — computed only when wall-clock
  budget remains, else known-gaps (feature value 0.0 + gap tag) per the
  OFFLINE-FIRST default in the integration plan
  (.hermes/plans/2026-08-26_174800-url-phishml-integration.md).

Model is UNPROVEN (0-star repo, no published eval) → the fusion trust bar
(AUC >= 0.75 on a known-benign/known-phish corpus) decides its weight; do not
raise it without re-proving on real data. Evidence: /tmp/phish_ml_status.md.