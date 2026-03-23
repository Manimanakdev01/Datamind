# -*- coding: utf-8 -*-
"""
DataMind AI v5 — Production AutoML + Deep Learning + NLP + Chatbot
Features: SHAP explainability · Firebase persistence · TensorFlow CNN/LSTM
          NLP text classification · Intent-based chatbot · Professional UI
"""

import io, os, pickle, warnings, datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap

from sklearn.calibration   import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble      import (GradientBoostingClassifier, GradientBoostingRegressor,
                                    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                    RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model  import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics       import (accuracy_score, classification_report, confusion_matrix,
                                    f1_score, mean_absolute_error, mean_squared_error,
                                    r2_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree          import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes   import MultinomialNB
from sklearn.svm           import LinearSVC

# ── TensorFlow / Keras ──
TF_AVAILABLE   = False
TF_IMPORT_ERR  = ""
KerasTokenizer = None
pad_sequences  = None
to_categorical = None
keras          = None
layers         = None

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    # Keras 3 (TF 2.16+) ships as standalone 'keras' package
    try:
        import keras as _keras_pkg
        keras  = _keras_pkg
        layers = _keras_pkg.layers
    except ImportError:
        keras  = tf.keras
        layers = tf.keras.layers

    # Preprocessing — moved in TF 2.12+; try multiple paths
    _tok, _pad, _cat = None, None, None

    # Path 1: legacy tensorflow.keras.preprocessing
    try:
        from tensorflow.keras.preprocessing.text     import Tokenizer as _T1
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _P1
        from tensorflow.keras.utils                  import to_categorical as _C1
        _tok, _pad, _cat = _T1, _P1, _C1
    except Exception:
        pass

    # Path 2: standalone keras.preprocessing (Keras 3)
    if _tok is None:
        try:
            from keras.preprocessing.text     import Tokenizer as _T2
            from keras.preprocessing.sequence import pad_sequences as _P2
            from keras.utils                  import to_categorical as _C2
            _tok, _pad, _cat = _T2, _P2, _C2
        except Exception:
            pass

    # Path 3: tf_keras compatibility shim
    if _tok is None:
        try:
            import tf_keras
            from tf_keras.preprocessing.text     import Tokenizer as _T3
            from tf_keras.preprocessing.sequence import pad_sequences as _P3
            from tf_keras.utils                  import to_categorical as _C3
            _tok, _pad, _cat = _T3, _P3, _C3
        except Exception:
            pass

    if _tok is None:
        raise ImportError(
            "Could not import Keras preprocessing. "
            "Run: pip install tf-keras  (for TF 2.16+) or downgrade to TF ≤ 2.15"
        )

    KerasTokenizer = _tok
    pad_sequences  = _pad
    to_categorical = _cat
    TF_AVAILABLE   = True

except Exception as _tf_err:
    TF_IMPORT_ERR = str(_tf_err)


# ── Firebase config — fill in YOUR project values ───────────────────────────
_FB_CONFIG = {
    "apiKey":            "AIzaSyAi1ASpJvlDS1JmTXoFMFsOK4XK_U5IepE",
    "authDomain":        "roboai-f189f.firebaseapp.com",
    "databaseURL":       "https://roboai-f189f-default-rtdb.firebaseio.com",
    "projectId":         "roboai-f189f",
    "storageBucket":     "roboai-f189f.firebasestorage.app",
    "messagingSenderId": "723351783147",
    "appId":             "1:723351783147:web:249775912d441a93c220ae",
}
_FB_SERVICE_ACCOUNT = "roboai-f189f-firebase-adminsdk-fbsvc-33a03916a8.json"

FIREBASE_AVAILABLE = False   # True when pyrebase auth is ready
RTDB_OK            = False   # True when Realtime DB is ready
fb_auth_obj        = None    # pyrebase auth
fb_db              = None    # pyrebase realtime database

try:
    import pyrebase as _pyrebase
    _fb_app     = _pyrebase.initialize_app(_FB_CONFIG)
    fb_auth_obj = _fb_app.auth()
    fb_db       = _fb_app.database()
    FIREBASE_AVAILABLE = True
    RTDB_OK            = True
except Exception:
    pass

# ── App constants ────────────────────────────────────────────────────────────
PLAN_PRICE     = 300
PLAN_FREE_PROJ = 2
UPI_ID         = "8789390866@ybl"
UPI_NAME       = "DataMind AI"
UPI_NOTE       = "DataMindAI Premium 1 Month"

# ── YOLO FastAPI microservice URL ─────────────────────────────────────────────
# Deploy yolo_api.py on Railway/Render/VPS, paste the public URL here.
# Or set env var: export YOLO_API_URL="https://your-api.railway.app"
YOLO_API_URL = os.environ.get("YOLO_API_URL", "")

# ── Gemini API key ───────────────────────────────────────────────────────────
# Free key: aistudio.google.com/app/apikey
# Add to Streamlit secrets: GEMINI_API_KEY = "AIza..."
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Razorpay — fill in YOUR values from razorpay.com/app/keys ────────────────
# Use rzp_test_... keys for sandbox, rzp_live_... for production
RAZORPAY_KEY_ID     = "rzp_live_XXXXXXXXXXXXXXXX"   # ← replace
RAZORPAY_KEY_SECRET = "XXXXXXXXXXXXXXXXXXXXXXXX"    # ← replace

warnings.filterwarnings("ignore")
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + THEME
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="DataMind AI", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root{
  --bg:#f0f2f5;--surface:#fff;--surface2:#f8f9fb;
  --sidebar:#0f1b2d;--border:#e2e6ea;
  --accent:#1a56db;--accent-lt:#e8effc;
  --gold:#b45309;--gold-lt:#fef3c7;
  --success:#047857;--success-lt:#d1fae5;
  --danger:#b91c1c;--danger-lt:#fee2e2;
  --text:#111827;--text2:#374151;--muted:#6b7280;
  --radius:8px;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  --shadow-md:0 4px 6px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.05);
}
html,body,[class*="css"]{background:#f0f2f5!important;color:var(--text)!important;font-family:'IBM Plex Sans',sans-serif!important;font-size:14px;}
section[data-testid="stSidebar"]{background:var(--sidebar)!important;border-right:none!important;}
section[data-testid="stSidebar"] *{color:#cbd5e1!important;}
section[data-testid="stSidebar"] .stRadio label{font-family:'IBM Plex Mono',monospace!important;font-size:0.72rem!important;letter-spacing:0.06em;text-transform:uppercase;}
section[data-testid="stSidebar"] hr{border-color:#1e3a5f!important;}
h1{font-size:1.65rem!important;font-weight:700!important;color:var(--text)!important;letter-spacing:-0.01em;}
h2{font-size:1.1rem!important;font-weight:600!important;color:var(--text)!important;}
h3{font-size:0.95rem!important;font-weight:600!important;color:var(--text2)!important;}
.dm-pagehead{background:var(--surface);border-bottom:1px solid var(--border);padding:1rem 1.5rem;margin:-1rem -1rem 1.5rem;display:flex;align-items:center;gap:0.75rem;box-shadow:var(--shadow);}
.dm-pagehead .icon{width:34px;height:34px;border-radius:8px;background:var(--accent-lt);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:700;}
.dm-pagehead .title{font-size:1rem;font-weight:600;color:var(--text);}
.dm-pagehead .sub{font-size:0.75rem;color:var(--muted);margin-top:1px;}
.dm-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem 1.5rem;margin-bottom:1rem;box-shadow:var(--shadow);}
.dm-card-title{font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:var(--muted);margin-bottom:0.85rem;padding-bottom:0.6rem;border-bottom:1px solid var(--border);}
.dm-kpi{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--accent);border-radius:var(--radius);padding:1rem 1.25rem;box-shadow:var(--shadow);}
.dm-kpi .val{font-family:'IBM Plex Mono',monospace;font-size:1.7rem;font-weight:600;color:var(--accent);}
.dm-kpi .lbl{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin-top:2px;}
.dm-kpi.gold{border-top-color:var(--gold);}.dm-kpi.gold .val{color:var(--gold);}
.dm-kpi.green{border-top-color:var(--success);}.dm-kpi.green .val{color:var(--success);}
.dm-kpi.red{border-top-color:var(--danger);}.dm-kpi.red .val{color:var(--danger);}
.dm-badge{display:inline-flex;align-items:center;gap:4px;padding:0.2rem 0.65rem;border-radius:20px;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:600;letter-spacing:0.04em;}
.dm-badge.blue{background:var(--accent-lt);color:var(--accent);}
.dm-badge.green{background:var(--success-lt);color:var(--success);}
.dm-badge.gold{background:var(--gold-lt);color:var(--gold);}
.dm-badge.red{background:var(--danger-lt);color:var(--danger);}
.dm-badge.grey{background:#f3f4f6;color:var(--muted);}
.dm-upload{background:var(--surface2);border:2px dashed var(--border);border-radius:var(--radius);padding:2.5rem;text-align:center;color:var(--muted);}
.dm-upload .uicon{font-size:2rem;margin-bottom:0.5rem;}
.dm-upload .hint{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;}
.dm-divider{height:1px;background:var(--border);margin:1.5rem 0;}
.dm-logo{padding:1.25rem 1rem 1rem;border-bottom:1px solid #1e3a5f;margin-bottom:0.5rem;}
.dm-logo .name{font-size:1.1rem;font-weight:700;color:#f8fafc;letter-spacing:-0.01em;}
.dm-logo .ver{font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#64748b;margin-top:2px;letter-spacing:0.08em;text-transform:uppercase;}
.dm-logo .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#22c55e;margin-right:5px;vertical-align:middle;}
.stDataFrame{border-radius:var(--radius)!important;box-shadow:var(--shadow)!important;}
.stButton>button{background:var(--accent)!important;color:#fff!important;font-family:'IBM Plex Sans',sans-serif!important;font-weight:500!important;font-size:0.82rem!important;border:none!important;border-radius:var(--radius)!important;padding:0.5rem 1.25rem!important;box-shadow:var(--shadow)!important;transition:all .15s!important;}
.stButton>button:hover{background:#1447c0!important;}
.stDownloadButton>button{background:var(--surface)!important;color:var(--accent)!important;border:1px solid var(--accent)!important;font-family:'IBM Plex Sans',sans-serif!important;font-size:0.82rem!important;border-radius:var(--radius)!important;}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:var(--surface)!important;border-color:var(--border)!important;border-radius:var(--radius)!important;color:var(--text)!important;}
button[data-baseweb="tab"]{font-family:'IBM Plex Sans',sans-serif!important;font-size:0.78rem!important;font-weight:500!important;color:var(--muted)!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;font-weight:600!important;}
.stAlert{border-radius:var(--radius)!important;font-size:0.82rem!important;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#ffffff","axes.facecolor":"#ffffff",
    "axes.edgecolor":"#e2e6ea","axes.labelcolor":"#374151",
    "xtick.color":"#6b7280","ytick.color":"#6b7280",
    "text.color":"#374151","grid.color":"#f3f4f6","grid.linestyle":"--",
    "font.family":"sans-serif","axes.titlesize":11,"axes.labelsize":9,
    "axes.titleweight":"600","axes.spines.top":False,"axes.spines.right":False,
})
C_BLUE="#1a56db"; C_GOLD="#b45309"; C_GREEN="#047857"
C_RED="#b91c1c";  C_PURPLE="#6d28d9"; C_SLATE="#475569"

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
for k,v in [("history",[]),("model",None),("features",[]),("train_df",None),
            ("problem",None),("label_encoders",{}),("y_test",None),("preds",None),
            ("proba",None),("col_meta",{}),("shap_values",None),("shap_X",None),
            ("dl_model",None),("dl_history",None),("dl_type",None),
            ("dl_tokenizer",None),("dl_classes",None),("dl_img_size",(64,64)),
            ("nlp_model",None),("nlp_vectorizer",None),("nlp_classes",None),
            ("chatbot_model",None),("chatbot_vectorizer",None),
            ("chatbot_responses",{}),("chatbot_classes",[]),("chat_history",[]),
            ("cluster_model",None),("cluster_labels",None),("cluster_df",None)]:
    if k not in st.session_state: st.session_state[k] = v

for _k,_v in [
    ("auth_uid",""),("auth_email",""),("auth_plan","free"),
    ("auth_is_admin",False),("auth_proj_used",0),
    ("auth_paid_until",None),("auth_page","login"),("auth_error",""),
    ("auth_token",""),          # Firebase idToken — required for authenticated RTDB writes
    ("auth_refresh_token",""),   # Firebase refreshToken — used to get new idToken
    ("auth_token_uid",""),       # uid the current token belongs to
]:
    if _k not in st.session_state: st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
#  REALTIME DATABASE HELPERS
#  Path: datamind_users/{uid}/
#  Fields: email, plan, admin, proj_used, paid_until, created_at,
#          last_login, last_utr, payment_ts
# ════════════════════════════════════════════════════════════════════════════

def _get_token():
    """Return current user idToken from session, or None."""
    return st.session_state.get("auth_token") or None

def _refresh_token():
    """
    Pyrebase tokens expire after 1 hour.
    Call this before any write to get a fresh token.
    Returns the token string or None.
    """
    if not FIREBASE_AVAILABLE:
        return None
    try:
        uid = st.session_state.get("auth_token_uid","")
        refresh = st.session_state.get("auth_refresh_token","")
        if not refresh:
            return _get_token()
        user = fb_auth_obj.refresh(refresh)
        token = user.get("idToken","")
        st.session_state.auth_token         = token
        st.session_state.auth_token_uid     = user.get("userId","")
        st.session_state.auth_refresh_token = user.get("refreshToken","")
        return token
    except Exception:
        return _get_token()

def _rtdb_get_user(uid):
    """Read user record — no token needed for reads if rules allow."""
    if not RTDB_OK or not uid:
        return None
    try:
        token = _get_token()
        if token:
            result = fb_db.child("datamind_users").child(uid).get(token)
        else:
            result = fb_db.child("datamind_users").child(uid).get()
        return result.val()
    except Exception:
        return None

def _rtdb_set_user(uid, data):
    """Create/overwrite a user record — requires auth token."""
    if not RTDB_OK or not uid:
        return
    try:
        token = _refresh_token()
        if token:
            fb_db.child("datamind_users").child(uid).set(data, token)
        else:
            fb_db.child("datamind_users").child(uid).set(data)
    except Exception as _e:
        pass

def _rtdb_update_user(uid, data):
    """
    Patch specific fields on a user record.
    Passes idToken so Firebase security rules allow authenticated writes.
    """
    if not RTDB_OK or not uid:
        return
    # Convert Python booleans explicitly (pyrebase serialises them correctly,
    # but being explicit avoids any edge-case with older versions)
    _safe = {}
    for k, v in data.items():
        _safe[k] = bool(v) if isinstance(v, bool) else v
    try:
        token = _refresh_token()
        if token:
            fb_db.child("datamind_users").child(uid).update(_safe, token)
        else:
            fb_db.child("datamind_users").child(uid).update(_safe)
    except Exception as _e:
        pass

def _rtdb_get_all_users():
    """
    Read ALL user records (admin only).
    Uses admin token so security rules must allow admin reads,
    OR set your rules to allow read for authenticated users.
    """
    if not RTDB_OK:
        return []
    try:
        token = _get_token()
        if token:
            result = fb_db.child("datamind_users").get(token)
        else:
            result = fb_db.child("datamind_users").get()
        val = result.val()
        if not val:
            return []
        users = []
        for uid, data in val.items():
            if isinstance(data, dict):
                data["_uid"] = uid
                users.append(data)
        return users
    except Exception:
        return []

def _sync_user_state(uid):
    doc = _rtdb_get_user(uid)
    if not doc:
        _rtdb_set_user(uid, {
            "email":      st.session_state.auth_email,
            "plan":       "free",
            "admin":      False,
            "proj_used":  0,
            "paid_until": "",
            "created_at": datetime.datetime.utcnow().isoformat(),
            "last_login": datetime.datetime.utcnow().isoformat(),
            "last_utr":   "",
        })
        st.session_state.auth_plan      = "free"
        st.session_state.auth_is_admin  = False
        st.session_state.auth_proj_used = 0
        st.session_state.auth_paid_until = None
        return

    is_admin  = bool(doc.get("admin", False))
    plan      = str(doc.get("plan", "free"))
    proj_used = int(doc.get("proj_used", 0))
    paid_str  = str(doc.get("paid_until", "") or "")

    st.session_state.auth_is_admin  = is_admin
    st.session_state.auth_proj_used = proj_used
    _rtdb_update_user(uid, {"last_login": datetime.datetime.utcnow().isoformat()})

    if is_admin:
        st.session_state.auth_plan = "premium"
        if not paid_str:
            _pu = (datetime.datetime.utcnow()+datetime.timedelta(days=365)).isoformat()
            _rtdb_update_user(uid, {"plan":"premium","paid_until":_pu})
            paid_str = _pu
        try:
            st.session_state.auth_paid_until = datetime.datetime.fromisoformat(paid_str)
        except Exception:
            st.session_state.auth_paid_until = None
        return

    # pending_review: payment submitted but not yet approved — keep state as-is
    if plan == "pending_review":
        st.session_state.auth_plan       = "pending_review"
        st.session_state.auth_paid_until = None
        return

    if paid_str:
        try:
            _dt = datetime.datetime.fromisoformat(paid_str)
            st.session_state.auth_paid_until = _dt
            if _dt >= datetime.datetime.utcnow():
                st.session_state.auth_plan = "premium"
            else:
                st.session_state.auth_plan = "free"
                _rtdb_update_user(uid, {"plan": "free"})
        except Exception:
            st.session_state.auth_plan       = plan
            st.session_state.auth_paid_until = None
    else:
        st.session_state.auth_plan       = plan
        st.session_state.auth_paid_until = None

def _do_login(email, password):
    st.session_state.auth_error = ""
    if not FIREBASE_AVAILABLE:
        st.session_state.auth_error = "Firebase not configured. Fill in _FB_CONFIG."
        return
    if not email.strip() or not password.strip():
        st.session_state.auth_error = "Email and password are required."
        return
    try:
        user = fb_auth_obj.sign_in_with_email_and_password(email.strip(), password)
        uid  = user["localId"]
        st.session_state.auth_uid           = uid
        st.session_state.auth_email         = email.strip()
        st.session_state.auth_token         = user.get("idToken", "")
        st.session_state.auth_refresh_token = user.get("refreshToken", "")
        st.session_state.auth_token_uid     = uid
        _sync_user_state(uid)
        st.session_state.auth_page  = "app"
    except Exception as e:
        msg = str(e)
        if "INVALID_PASSWORD" in msg or "INVALID_LOGIN_CREDENTIALS" in msg:
            st.session_state.auth_error = "Incorrect email or password."
        elif "EMAIL_NOT_FOUND" in msg or "USER_NOT_FOUND" in msg:
            st.session_state.auth_error = "No account found. Please sign up."
        elif "TOO_MANY_ATTEMPTS" in msg:
            st.session_state.auth_error = "Too many attempts. Try again later."
        else:
            st.session_state.auth_error = "Login failed. Try again."

def _do_signup(email, password, confirm):
    st.session_state.auth_error = ""
    if not FIREBASE_AVAILABLE:
        st.session_state.auth_error = "Firebase not configured. Fill in _FB_CONFIG."
        return
    if not email.strip():
        st.session_state.auth_error = "Email is required."
        return
    if password != confirm:
        st.session_state.auth_error = "Passwords do not match."
        return
    if len(password) < 6:
        st.session_state.auth_error = "Password must be at least 6 characters."
        return
    try:
        user = fb_auth_obj.create_user_with_email_and_password(email.strip(), password)
        uid  = user["localId"]
        st.session_state.auth_uid           = uid
        st.session_state.auth_email         = email.strip()
        st.session_state.auth_token         = user.get("idToken", "")
        st.session_state.auth_refresh_token = user.get("refreshToken", "")
        st.session_state.auth_token_uid     = uid
        _sync_user_state(uid)
        st.session_state.auth_page  = "app"
    except Exception as e:
        msg = str(e)
        if "EMAIL_EXISTS" in msg:
            st.session_state.auth_error = "Email already registered. Please sign in."
        elif "WEAK_PASSWORD" in msg:
            st.session_state.auth_error = "Password too weak (min 6 chars)."
        elif "INVALID_EMAIL" in msg:
            st.session_state.auth_error = "Invalid email address."
        else:
            st.session_state.auth_error = "Signup failed: " + msg[:80]

def _do_logout():
    st.session_state.auth_uid           = ""
    st.session_state.auth_email         = ""
    st.session_state.auth_plan          = "free"
    st.session_state.auth_is_admin      = False
    st.session_state.auth_proj_used     = 0
    st.session_state.auth_paid_until    = None
    st.session_state.auth_token         = ""
    st.session_state.auth_refresh_token = ""
    st.session_state.auth_token_uid     = ""
    st.session_state.auth_error         = ""
    st.session_state.auth_page          = "login"

def _validate_utr(utr: str):
    """
    Returns (True, "") if UTR is valid, else (False, error_message).
    Rules:
      1. Must be exactly 12 digits (IMPS/UPI UTR standard).
      2. Must not already exist on any other account (prevents UTR reuse).
    """
    utr = utr.strip()
    if not utr:
        return False, "Please enter your UTR / Transaction ID."
    if not utr.isdigit():
        return False, "UTR must contain digits only — no letters or spaces."
    if len(utr) != 12:
        return False, f"UTR must be exactly 12 digits (you entered {len(utr)})."
    # Duplicate check — scan all users for this UTR
    all_users = _rtdb_get_all_users()
    for _u in all_users:
        if _u.get("last_utr", "").strip() == utr:
            return False, (
                "This UTR has already been used on another account. "
                "If you believe this is an error, contact support."
            )
    return True, ""


def _submit_payment(uid, utr):
    """
    Submit a payment for admin review.
    Sets plan to 'pending_review' — admin must approve from the Admin panel.
    Returns (True, message) on success, (False, error) on failure.
    """
    utr = utr.strip()
    ok, err = _validate_utr(utr)
    if not ok:
        return False, err
    _rtdb_update_user(uid, {
        "plan":       "pending_review",
        "last_utr":   utr,
        "payment_ts": datetime.datetime.utcnow().isoformat(),
    })
    return True, "Payment submitted! Your account will be activated within 24 hours after admin verification."

def _increment_project(uid):
    if st.session_state.auth_plan == "premium":
        return True
    used = st.session_state.auth_proj_used
    if used >= PLAN_FREE_PROJ:
        return False
    st.session_state.auth_proj_used = used + 1
    _rtdb_update_user(uid, {"proj_used": used+1})
    return True

def _require_premium(feature_name="this feature"):
    if st.session_state.auth_plan == "premium":
        return True
    if st.session_state.auth_plan == "pending_review":
        st.markdown(
            '<div class="dm-card" style="border-top:4px solid #d97706;'
            'background:#fffbeb;text-align:center;padding:2rem">'
            '<div style="font-size:2rem;margin-bottom:.5rem">⏳</div>'
            '<div style="font-size:1rem;font-weight:700;color:#111827;margin-bottom:.4rem">'
            'Payment Under Review</div>'
            '<div style="font-size:0.85rem;color:#374151;">'
            'Your payment is being verified by our team.<br>'
            '<strong>' + feature_name + '</strong> will unlock automatically '
            'once your UTR is approved (within 24 hours).'
            '</div></div>', unsafe_allow_html=True)
        return False
    st.markdown(
        '<div class="dm-card" style="border-top:4px solid #b45309;'
        'background:#fffbeb;text-align:center;padding:2rem">'
        '<div style="font-size:2rem;margin-bottom:.5rem">&#128274;</div>'
        '<div style="font-size:1rem;font-weight:700;color:#111827;margin-bottom:.4rem">'
        'Premium Feature</div>'
        '<div style="font-size:0.85rem;color:#374151;margin-bottom:1.2rem">'
        '<strong>'+feature_name+'</strong> is available on Premium only.<br>'
        'Upgrade for &#8377;'+str(PLAN_PRICE)+'/month to unlock everything.'
        '</div>'
        '<ul style="display:inline-block;text-align:left;font-size:0.82rem;'
        'line-height:2.2;color:#374151;margin-bottom:1.2rem">'
        '<li>&#10003; Unlimited projects</li>'
        '<li>&#10003; Deep Learning &middot; CNN &middot; LSTM</li>'
        '<li>&#10003; Ollama hybrid auto-labeling</li>'
        '<li>&#10003; NLP &amp; Chatbot training</li>'
        '<li>&#10003; SHAP explainability</li>'
        '</ul></div>', unsafe_allow_html=True)
    if st.button("Upgrade to Premium  Rs."+str(PLAN_PRICE)+"/mo",
                 key="upg_"+feature_name[:8].replace(" ","_"),
                 use_container_width=True):
        st.session_state.auth_page = "payment"; st.rerun()
    return False

def _generate_upi_qr():
    try:
        import qrcode as _qc
        _upi = (f"upi://pay?pa={UPI_ID}&pn={UPI_NAME.replace(' ','%20')}"
                f"&am={PLAN_PRICE}&cu=INR&tn={UPI_NOTE.replace(' ','%20')}")
        _q = _qc.QRCode(version=2, box_size=8, border=3)
        _q.add_data(_upi); _q.make(fit=True)
        return _q.make_image(fill_color="#1a56db", back_color="white")
    except Exception:
        return None

# ── Razorpay helpers ─────────────────────────────────────────────────────────

def _razorpay_create_order(amount_inr=PLAN_PRICE):
    import requests as _rq, uuid as _uuid
    _resp = _rq.post(
        "https://api.razorpay.com/v1/orders",
        json={
            "amount":   amount_inr * 100,
            "currency": "INR",
            "receipt":  "dm_" + str(_uuid.uuid4())[:12],
            "notes":    {"product": UPI_NAME, "plan": "premium_1month"},
        },
        auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
        timeout=10,
    )
    _resp.raise_for_status()
    return _resp.json()


def _razorpay_verify_signature(order_id, payment_id, signature):
    import hmac as _hmac, hashlib as _hl
    _msg  = (order_id + "|" + payment_id).encode("utf-8")
    _key  = RAZORPAY_KEY_SECRET.encode("utf-8")
    _exp  = _hmac.new(_key, _msg, _hl.sha256).hexdigest()
    return _hmac.compare_digest(_exp, signature)


def _activate_razorpay_payment(uid, payment_id, order_id):
    _pu = (datetime.datetime.utcnow() + datetime.timedelta(days=30)).isoformat()
    _rtdb_update_user(uid, {
        "plan":              "premium",
        "paid_until":        _pu,
        "proj_used":         0,
        "last_utr":          payment_id,
        "payment_ts":        datetime.datetime.utcnow().isoformat(),
        "payment_method":    "razorpay",
        "razorpay_order_id": order_id,
    })
    st.session_state.auth_plan       = "premium"
    st.session_state.auth_proj_used  = 0
    try:
        st.session_state.auth_paid_until = datetime.datetime.fromisoformat(_pu)
    except Exception:
        pass


def _render_razorpay_button(order_id, amount_inr, user_email):
    import streamlit.components.v1 as _comp
    _html = (
        "<style>"
        "#rzp-btn{background:#1a56db;color:#fff;border:none;border-radius:8px;"
        "padding:13px 0;font-size:15px;font-weight:700;cursor:pointer;width:100%;"
        "font-family:IBM Plex Sans,sans-serif;"
        "box-shadow:0 4px 14px rgba(26,86,219,.35);transition:background .15s}"
        "#rzp-btn:hover{background:#1648c0}"
        "#rzp-status{margin-top:10px;font-size:13px;color:#374151;text-align:center}"
        "</style>"
        "<script src='https://checkout.razorpay.com/v1/checkout.js'></script>"
        "<button id='rzp-btn' onclick='openRazorpay()'>"
        "Pay &#8377;" + str(amount_inr) + " securely via Razorpay"
        "</button>"
        "<div id='rzp-status'></div>"
        "<script>"
        "function openRazorpay(){"
          "var o={"
            "key:'" + RAZORPAY_KEY_ID + "',"
            "amount:" + str(amount_inr * 100) + ","
            "currency:'INR',"
            "name:'" + UPI_NAME + "',"
            "description:'Premium Plan - 30 days',"
            "order_id:'" + order_id + "',"
            "prefill:{email:'" + user_email + "'},"
            "theme:{color:'#1a56db'},"
            "modal:{ondismiss:function(){"
              "document.getElementById('rzp-status').innerText="
              "'Payment window closed. Click to retry.';}},"
            "handler:function(r){"
              "document.getElementById('rzp-status').innerText='Verifying...';"
              "var u=new URL(window.parent.location.href);"
              "u.searchParams.set('rzp_payment_id',r.razorpay_payment_id);"
              "u.searchParams.set('rzp_order_id',r.razorpay_order_id);"
              "u.searchParams.set('rzp_signature',r.razorpay_signature);"
              "window.parent.location.href=u.toString();}};"
          "var rzp=new Razorpay(o);"
          "rzp.on('payment.failed',function(e){"
            "document.getElementById('rzp-status').innerText="
            "'Payment failed: '+e.error.description;});"
          "rzp.open();}"
        "window.onload=function(){openRazorpay();};"
        "</script>"
    )
    _comp.html(_html, height=130, scrolling=False)


def _check_razorpay_callback():
    _qp     = st.query_params
    _pay_id = _qp.get("rzp_payment_id", "")
    _ord_id = _qp.get("rzp_order_id",   "")
    _sig    = _qp.get("rzp_signature",  "")
    if not (_pay_id and _ord_id and _sig):
        return False
    st.query_params.clear()
    uid = st.session_state.get("auth_uid", "")
    if not uid:
        return False
    if _razorpay_verify_signature(_ord_id, _pay_id, _sig):
        _activate_razorpay_payment(uid, _pay_id, _ord_id)
        return True
    st.error("Payment signature verification failed. Contact support with Payment ID: " + _pay_id)
    return False



# ════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_csv(b):
    return pd.read_csv(io.BytesIO(b))

def feature_engineering(df):
    df=df.copy(); encoders={}
    for col in list(df.columns):
        if "date" in col.lower():
            try:
                df[col]=pd.to_datetime(df[col])
                for part,fn in [("year",lambda x:x.dt.year),("month",lambda x:x.dt.month),
                                 ("day",lambda x:x.dt.day),("dow",lambda x:x.dt.dayofweek)]:
                    df[f"{col}_{part}"]=fn(df[col])
                df.drop(columns=[col],inplace=True); continue
            except: pass
        if df[col].dtype=="object":
            try: df[col]=df[col].astype(float)
            except:
                le=LabelEncoder(); df[col]=le.fit_transform(df[col].astype(str)); encoders[col]=le
        if np.issubdtype(df[col].dtype,np.number):
            if df[col].skew()>1 and df[col].min()>=0: df[col]=np.log1p(df[col])
    return df.fillna(df.median(numeric_only=True)), encoders

@st.cache_data(show_spinner=False)
def cached_feature_engineering(b):
    return feature_engineering(load_csv(b))

def detect_leakage(X,y,thr=0.97):
    return [(c,round(abs(np.corrcoef(X[c].values,y.values)[0,1]),3))
            for c in X.select_dtypes(include=np.number).columns
            if abs(np.corrcoef(X[c].values,y.values)[0,1])>thr]

def fairness_check(y_true,y_pred):
    return {str(v):round(float((y_pred[y_true==v]==y_true[y_true==v]).mean()),3)
            for v in np.unique(y_true)}

def detect_drift(train_df,new_df):
    return [c for c in train_df.columns
            if c in new_df.columns and np.issubdtype(train_df[c].dtype,np.number)
            and train_df[c].std()>0
            and abs(train_df[c].mean()-new_df[c].mean())>train_df[c].std()]

def fig_to_bytes(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",
                                   facecolor=fig.get_facecolor()); buf.seek(0); return buf.read()

def encode_inputs(raw_inputs,features,label_encoders):
    enc=[]
    for col in features:
        v=raw_inputs[col]
        if col in label_encoders:
            le=label_encoders[col]; v=le.transform([v])[0] if v in le.classes_ else 0
        enc.append(float(v))
    return np.array(enc).reshape(1,-1)

# ══════════════════════════════════════════════════════════
#  SHAP
# ══════════════════════════════════════════════════════════
def compute_shap(model, X_train, X_test, features):
    try:
        inner=model
        if hasattr(inner,"base_estimator"): inner=inner.base_estimator
        X_tr2,X_te2=X_train.copy(),X_test.copy()
        if hasattr(inner,"named_steps"):
            scaler=inner.named_steps.get("scaler",None)
            inner=inner.named_steps.get("model",inner)
            if scaler:
                X_tr2=pd.DataFrame(scaler.transform(X_train),columns=features)
                X_te2=pd.DataFrame(scaler.transform(X_test), columns=features)
        if hasattr(inner,"feature_importances_"):
            exp=shap.TreeExplainer(inner); sv=exp.shap_values(X_te2)
        else:
            exp=shap.LinearExplainer(inner,X_tr2); sv=exp.shap_values(X_te2)
        if isinstance(sv,list):
            sv=sv[1] if len(sv)==2 else np.mean(np.abs(sv),axis=0)
        return np.array(sv)
    except: return None

@st.cache_data(show_spinner=False)
def _plot_shap_bar(sv_b,feat_b,feature_names):
    sv=np.frombuffer(sv_b,dtype=np.float64).reshape(-1,len(feature_names))
    ma=np.abs(sv).mean(axis=0); idx=np.argsort(ma)[-15:]
    fig,ax=plt.subplots(figsize=(7,max(3,len(idx)*0.4)))
    bars=ax.barh([feature_names[i] for i in idx],[ma[i] for i in idx],
                 color=C_BLUE,alpha=0.82,edgecolor="white",linewidth=0.5)
    ax.bar_label(bars,fmt="%.4f",padding=4,fontsize=7.5,color=C_SLATE)
    ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Feature Impact (SHAP)")
    ax.set_xlim(0,ma[idx].max()*1.18); fig.tight_layout()
    d=fig_to_bytes(fig); plt.close(fig); return d

@st.cache_data(show_spinner=False)
def _plot_shap_beeswarm(sv_b,X_b,feature_names):
    sv=np.frombuffer(sv_b,dtype=np.float64).reshape(-1,len(feature_names))
    X =np.frombuffer(X_b, dtype=np.float64).reshape(-1,len(feature_names))
    ma=np.abs(sv).mean(axis=0); idx=np.argsort(ma)[-12:]
    sv2=sv[:,idx]; X2=X[:,idx]; fn2=[feature_names[i] for i in idx]
    fig,ax=plt.subplots(figsize=(8,max(4,len(idx)*0.48)))
    for j,fn in enumerate(fn2):
        jit=np.random.uniform(-0.18,0.18,len(sv2))
        sc=ax.scatter(sv2[:,j],j+jit,c=X2[:,j],cmap="RdBu_r",
                      alpha=0.5,s=12,vmin=X2[:,j].min(),vmax=X2[:,j].max())
    ax.set_yticks(range(len(fn2))); ax.set_yticklabels(fn2,fontsize=8)
    ax.axvline(0,color="#9ca3af",lw=1,linestyle="--")
    ax.set_xlabel("SHAP value"); ax.set_title("SHAP Beeswarm — feature effect direction")
    plt.colorbar(sc,ax=ax,label="Feature value",shrink=0.6)
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

# ══════════════════════════════════════════════════════════
#  CACHED EVAL PLOTS
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _plot_cm(yt,yp):
    y1=np.frombuffer(yt,dtype=np.float64); y2=np.frombuffer(yp,dtype=np.float64)
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(confusion_matrix(y1,y2),annot=True,fmt="d",cmap="Blues",
                linewidths=0.5,linecolor="#e2e6ea",ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

@st.cache_data(show_spinner=False)
def _plot_roc(yt,pb):
    y1=np.frombuffer(yt,dtype=np.float64); p=np.frombuffer(pb,dtype=np.float64)
    fig,ax=plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_predictions(y1,p,ax=ax,color=C_BLUE,lw=2)
    ax.plot([0,1],[0,1],"--",color="#9ca3af",lw=1); ax.set_title("ROC Curve")
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

@st.cache_data(show_spinner=False)
def _plot_cal(yt,pb):
    y1=np.frombuffer(yt,dtype=np.float64); p=np.frombuffer(pb,dtype=np.float64)
    fp,mp=calibration_curve(y1,p,n_bins=10)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(mp,fp,"s-",color=C_BLUE,label="Model",lw=2)
    ax.plot([0,1],[0,1],"--",color=C_GOLD,label="Perfect",lw=1.5)
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve"); ax.legend()
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

@st.cache_data(show_spinner=False)
def _plot_res(yt,yp):
    y1=np.frombuffer(yt,dtype=np.float64); y2=np.frombuffer(yp,dtype=np.float64)
    res=y1-y2
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(y2,res,alpha=0.45,color=C_BLUE,s=16,edgecolors="white",linewidth=0.3)
    axes[0].axhline(0,color=C_RED,lw=1.5,linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(res,bins=30,color=C_BLUE,alpha=0.75,edgecolor="white")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

@st.cache_data(show_spinner=False)
def _plot_avp(yt,yp):
    y1=np.frombuffer(yt,dtype=np.float64); y2=np.frombuffer(yp,dtype=np.float64)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.scatter(y1,y2,alpha=0.45,color=C_BLUE,s=16,edgecolors="white",linewidth=0.3)
    mn,mx=min(y1.min(),y2.min()),max(y1.max(),y2.max())
    ax.plot([mn,mx],[mn,mx],"--",color=C_GOLD,lw=1.5)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted")
    fig.tight_layout(); d=fig_to_bytes(fig); plt.close(fig); return d

# ════════════════════════════════════════════════════════════════════════════
#  AUTH WALL
# ════════════════════════════════════════════════════════════════════════════
_A = (
    "<style>"
    # ── Streamlit shell removal ──────────────────────────
    "[data-testid='stAppViewContainer']{padding:0!important;}"
    "[data-testid='stHeader']{display:none!important;}"
    "[data-testid='stToolbar']{display:none!important;}"
    "[data-testid='stDecoration']{display:none!important;}"
    "footer{display:none!important;}"
    "section[data-testid='stSidebar']{display:none!important;}"
    ".block-container{padding:0!important;max-width:100%!important;margin:0!important;}"
    "html,body,[data-testid='stAppViewContainer']{background:#050d1e!important;}"
    # ── Layout: two-column full-height ────────────────────
    ".aw{display:flex;min-height:100vh;font-family:'IBM Plex Sans',sans-serif;}"
    # ── Left hero panel ───────────────────────────────────
    ".al{flex:0 0 48%;position:relative;overflow:hidden;"
    "background:linear-gradient(145deg,#050d1e 0%,#0b1e45 50%,#1245a8 100%);"
    "display:flex;flex-direction:column;justify-content:center;padding:3rem 2.5rem;}"
    ".al-g1{position:absolute;width:380px;height:380px;border-radius:50%;"
    "background:radial-gradient(circle,rgba(59,130,246,.28) 0%,transparent 65%);"
    "top:-120px;right:-100px;pointer-events:none;}"
    ".al-g2{position:absolute;width:260px;height:260px;border-radius:50%;"
    "background:radial-gradient(circle,rgba(6,182,212,.22) 0%,transparent 65%);"
    "bottom:-80px;left:-60px;pointer-events:none;}"
    ".al-g3{position:absolute;width:200px;height:200px;border-radius:50%;"
    "background:radial-gradient(circle,rgba(139,92,246,.18) 0%,transparent 65%);"
    "top:40%;right:5%;pointer-events:none;}"
    ".al-logo{font-size:1.7rem;font-weight:800;color:#fff;letter-spacing:-.03em;"
    "margin-bottom:.2rem;position:relative;z-index:1;}"
    ".al-logo em{color:#60a5fa;font-style:normal;}"
    ".al-tag{font-size:.68rem;color:#7dd3fc;letter-spacing:.14em;"
    "text-transform:uppercase;margin-bottom:2.8rem;position:relative;z-index:1;}"
    ".al-f{display:flex;align-items:center;gap:.85rem;margin-bottom:1.25rem;"
    "position:relative;z-index:1;}"
    ".al-fi{width:38px;height:38px;border-radius:11px;flex-shrink:0;"
    "display:flex;align-items:center;justify-content:center;font-size:.95rem;"
    "background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);}"
    ".al-ft{font-size:.86rem;font-weight:600;color:#f0f9ff;line-height:1.25;}"
    ".al-fs{font-size:.7rem;color:#93c5fd;margin-top:2px;}"
    ".al-chips{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:2.4rem;"
    "position:relative;z-index:1;}"
    ".al-chip{padding:.2rem .7rem;border-radius:20px;font-size:.65rem;"
    "font-weight:600;background:rgba(255,255,255,.09);color:#bfdbfe;"
    "border:1px solid rgba(255,255,255,.14);}"
    ".al-pricebox{margin-top:2rem;background:rgba(255,255,255,.07);"
    "border:1px solid rgba(255,255,255,.13);border-radius:14px;"
    "padding:1.1rem 1.4rem;position:relative;z-index:1;}"
    ".al-pricebox .amount{font-size:2rem;font-weight:800;color:#fff;}"
    ".al-pricebox .per{font-size:.8rem;font-weight:400;color:#93c5fd;margin-left:.3rem;}"
    ".al-pricebox .sub{font-size:.7rem;color:#bfdbfe;margin-top:.2rem;}"
    # ── Right form panel ──────────────────────────────────
    ".ar{flex:1;background:#f0f4fc;display:flex;align-items:center;"
    "justify-content:center;padding:2rem 1.5rem;}"
    ".ac{width:100%;max-width:400px;background:#fff;border-radius:22px;"
    "padding:2.6rem 2.2rem;"
    "box-shadow:0 16px 56px rgba(8,16,50,.12),0 2px 8px rgba(8,16,50,.07);"
    "border:1px solid #e2eaf6;}"
    ".ac-logo{display:flex;align-items:center;gap:.6rem;margin-bottom:1.6rem;}"
    ".ac-logo-dot{width:32px;height:32px;border-radius:9px;"
    "background:linear-gradient(135deg,#1a56db,#2563eb);"
    "display:flex;align-items:center;justify-content:center;"
    "color:#fff;font-size:.9rem;font-weight:800;}"
    ".ac-logo-name{font-size:.9rem;font-weight:700;color:#0a1428;"
    "letter-spacing:-.01em;}"
    ".ac-logo-ver{font-size:.65rem;color:#94a3b8;}"
    ".ac-title{font-size:1.3rem;font-weight:700;color:#0a1428;"
    "letter-spacing:-.02em;margin-bottom:.25rem;}"
    ".ac-sub{font-size:.79rem;color:#64748b;margin-bottom:1.5rem;}"
    ".ac-chip{display:inline-flex;align-items:center;gap:.3rem;"
    "background:#eef4ff;border:1px solid #bdd3fc;border-radius:20px;"
    "padding:.22rem .75rem;font-size:.69rem;font-weight:600;color:#1a56db;"
    "margin-bottom:1.3rem;}"
    ".ac-err{display:flex;align-items:center;gap:.4rem;background:#fff1f1;"
    "border:1px solid #fca5a5;color:#b91c1c;border-radius:10px;"
    "padding:.55rem .85rem;font-size:.77rem;margin-bottom:.9rem;}"
    ".ac-foot{margin-top:1.5rem;padding-top:1.1rem;border-top:1px solid #f1f5f9;"
    "text-align:center;font-size:.68rem;color:#94a3b8;}"
    # ── Streamlit widget overrides ────────────────────────
    "div[data-testid='stTextInput']>div>div>input{"
    "border-radius:10px!important;border:1.5px solid #dde4f0!important;"
    "font-size:.86rem!important;padding:.6rem .95rem!important;"
    "background:#f7fafd!important;color:#0a1428!important;"
    "transition:border-color .18s,box-shadow .18s!important;}"
    "div[data-testid='stTextInput']>div>div>input:focus{"
    "border-color:#1a56db!important;background:#fff!important;"
    "box-shadow:0 0 0 3px rgba(26,86,219,.13)!important;}"
    "div[data-testid='stTextInput'] label{"
    "font-size:.69rem!important;font-weight:600!important;color:#374151!important;"
    "text-transform:uppercase!important;letter-spacing:.07em!important;}"
    ".ap>button{background:linear-gradient(135deg,#1a56db,#2563eb)!important;"
    "color:#fff!important;border:none!important;border-radius:10px!important;"
    "font-size:.86rem!important;font-weight:600!important;"
    "padding:.65rem 1.5rem!important;width:100%!important;"
    "box-shadow:0 4px 18px rgba(26,86,219,.38)!important;"
    "transition:all .18s!important;margin-top:.5rem!important;}"
    ".ap>button:hover{background:linear-gradient(135deg,#1447c0,#1d4ed8)!important;"
    "box-shadow:0 6px 24px rgba(26,86,219,.48)!important;"
    "transform:translateY(-1px)!important;}"
    ".ag>button{background:#fff!important;color:#1a56db!important;"
    "border:1.5px solid #d0daf0!important;border-radius:10px!important;"
    "font-size:.83rem!important;font-weight:500!important;width:100%!important;"
    "box-shadow:none!important;transition:all .18s!important;"
    "margin-top:.35rem!important;}"
    ".ag>button:hover{border-color:#1a56db!important;background:#f0f6ff!important;}"
    ".aor{display:flex;align-items:center;gap:.6rem;margin:.8rem 0;"
    "color:#94a3b8;font-size:.73rem;}"
    ".aor::before,.aor::after{content:'';flex:1;height:1px;background:#e8edf5;}"
    "</style>"
)

def _left_base(tagline, extra_html=""):
    return (
        '<div class="aw"><div class="al">'
        '<div class="al-g1"></div><div class="al-g2"></div><div class="al-g3"></div>'
        '<div class="al-logo">&#9672; DataMind <em>AI</em></div>'
        '<div class="al-tag">v5.0 &middot; '+tagline+'</div>'
        '<div class="al-f"><div class="al-fi">&#129302;</div>'
        '<div><div class="al-ft">AutoML in one click</div>'
        '<div class="al-fs">Train &amp; evaluate ML models instantly</div></div></div>'
        '<div class="al-f"><div class="al-fi">&#127991;</div>'
        '<div><div class="al-ft">Auto Labeling Studio</div>'
        '<div class="al-fs">Scale AI-style labeling + Ollama LLM</div></div></div>'
        '<div class="al-f"><div class="al-fi">&#129504;</div>'
        '<div><div class="al-ft">Deep Learning &middot; CNN &middot; LSTM</div>'
        '<div class="al-fs">TensorFlow models in seconds</div></div></div>'
        '<div class="al-f"><div class="al-fi">&#128202;</div>'
        '<div><div class="al-ft">SHAP Explainability</div>'
        '<div class="al-fs">Understand every prediction</div></div></div>'
        + extra_html +
        '<div class="al-chips">'
        '<span class="al-chip">Free &middot; 2 Projects</span>'
        '<span class="al-chip">Premium &middot; &#8377;300/mo</span>'
        '<span class="al-chip">Firebase Auth</span>'
        '<span class="al-chip">Realtime DB</span>'
        '</div>'
        '</div>'   # closes .al
    )

def _right_open():
    st.markdown(
        '<div class="ar"><div class="ac">'
        '<div class="ac-logo">'
        '<div class="ac-logo-dot">&#9672;</div>'
        '<div><div class="ac-logo-name">DataMind AI</div>'
        '<div class="ac-logo-ver">Production v5.0</div></div>'
        '</div>',
        unsafe_allow_html=True)

def _right_close():
    st.markdown('</div></div></div>', unsafe_allow_html=True)


if st.session_state.auth_page == "login":
    st.markdown(_A, unsafe_allow_html=True)
    _c1, _c2 = st.columns([1.1, 1])
    with _c1:
        st.markdown(_left_base("Sign in to continue"), unsafe_allow_html=True)
    with _c2:
        _right_open()
        st.markdown(
            '<div class="ac-title">Welcome back</div>'
            '<div class="ac-sub">Sign in to your DataMind AI account</div>',
            unsafe_allow_html=True)
        if st.session_state.auth_error:
            st.markdown(
                '<div class="ac-err">&#9888;&nbsp;'+str(st.session_state.auth_error)+'</div>',
                unsafe_allow_html=True)
        _em = st.text_input("Email address", key="li_em", placeholder="you@example.com")
        _pw = st.text_input("Password", type="password", key="li_pw", placeholder="Your password")
        st.markdown('<div class="ap">', unsafe_allow_html=True)
        if st.button("Sign In", key="li_btn", use_container_width=True):
            _do_login(_em, _pw); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="aor">or</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;font-size:.78rem;color:#64748b;margin-bottom:.3rem">'
            'New here? Create a free account</div>', unsafe_allow_html=True)
        st.markdown('<div class="ag">', unsafe_allow_html=True)
        if st.button("Create Account", key="li_signup", use_container_width=True):
            st.session_state.auth_page="signup"; st.session_state.auth_error=""; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ac-foot">Free plan &middot; 2 projects &middot;'
            ' No credit card needed</div>', unsafe_allow_html=True)
        _right_close()
    st.stop()

elif st.session_state.auth_page == "signup":
    st.markdown(_A, unsafe_allow_html=True)
    _c1, _c2 = st.columns([1.1, 1])
    with _c1:
        st.markdown(_left_base("Create your account"), unsafe_allow_html=True)
    with _c2:
        _right_open()
        st.markdown(
            '<div class="ac-title">Get started free</div>'
            '<div class="ac-sub">No credit card &middot; Cancel anytime</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ac-chip">&#10024; Free plan includes 2 projects</div>',
            unsafe_allow_html=True)
        if st.session_state.auth_error:
            st.markdown(
                '<div class="ac-err">&#9888;&nbsp;'+str(st.session_state.auth_error)+'</div>',
                unsafe_allow_html=True)
        _em2 = st.text_input("Email address",   key="su_em",  placeholder="you@example.com")
        _pw2 = st.text_input("Password",         type="password", key="su_pw", placeholder="Min 6 characters")
        _pw3 = st.text_input("Confirm password", type="password", key="su_pw2",placeholder="Repeat password")
        st.markdown('<div class="ap">', unsafe_allow_html=True)
        if st.button("Create Account", key="su_btn", use_container_width=True):
            _do_signup(_em2, _pw2, _pw3); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="aor">already have an account?</div>', unsafe_allow_html=True)
        st.markdown('<div class="ag">', unsafe_allow_html=True)
        if st.button("Sign in instead", key="su_login", use_container_width=True):
            st.session_state.auth_page="login"; st.session_state.auth_error=""; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ac-foot">By signing up you agree to our Terms of Service</div>',
            unsafe_allow_html=True)
        _right_close()
    st.stop()

elif st.session_state.auth_page == "payment":
    st.markdown(_A, unsafe_allow_html=True)
    _uid_pay = st.session_state.auth_uid

    # Check if Razorpay just redirected back with a payment result
    if _check_razorpay_callback():
        st.success("Payment verified! Premium activated for 30 days.")
        import time as _tz; _tz.sleep(1.5)
        st.session_state.pop("rzp_order", None)
        st.session_state.auth_page = "app"; st.rerun()

    _pricebox = (
        '<div class="al-pricebox">'
        '<div><span class="amount">&#8377;300</span>'
        '<span class="per">/ month</span></div>'
        '<div class="sub">30 days full access &middot; Renew anytime</div>'
        '</div>'
    )
    _c1, _c2 = st.columns([1.1, 1])
    with _c1:
        st.markdown(_left_base("Upgrade to Premium", extra_html=_pricebox),
                    unsafe_allow_html=True)
    with _c2:
        _right_open()
        st.markdown(
            '<div class="ac-title">Choose Payment Method</div>'
            '<div class="ac-sub">Instant via Razorpay &middot; or manual UPI transfer</div>',
            unsafe_allow_html=True)

        _ptab_rzp, _ptab_upi = st.tabs(["💳 Razorpay (Instant)", "🏦 UPI / Manual"])

        with _ptab_rzp:
            st.markdown(
                '<div style="background:#f0f7ff;border:1px solid #bdd3fc;border-radius:10px;'
                'padding:.7rem 1rem;font-size:.8rem;color:#1e40af;margin-bottom:.8rem">'
                '&#9889; Payment is verified automatically — '
                'Premium activates <strong>instantly</strong> after checkout.</div>',
                unsafe_allow_html=True)
            if "rzp_order" not in st.session_state:
                try:
                    st.session_state.rzp_order = _razorpay_create_order(PLAN_PRICE)
                except Exception as _e:
                    st.session_state.rzp_order = None
                    st.warning("Razorpay unavailable: " + str(_e) + ". Use UPI tab instead.")
            _rzp_o = st.session_state.get("rzp_order")
            if _rzp_o and _rzp_o.get("id"):
                _render_razorpay_button(
                    order_id=_rzp_o["id"],
                    amount_inr=PLAN_PRICE,
                    user_email=st.session_state.auth_email,
                )
            else:
                st.info("Fill in RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in main.py to enable this.")

        with _ptab_upi:
            _qri = _generate_upi_qr()
            if _qri:
                import io as _bio; _buf = _bio.BytesIO(); _qri.save(_buf, format="PNG")
                _q1, _q2, _q3 = st.columns([1, 2, 1])
                with _q2:
                    st.image(_buf.getvalue(), use_container_width=True)
            st.markdown(
                '<div style="background:#eef4ff;border:1px solid #bdd3fc;border-radius:12px;'
                'padding:.8rem 1rem;margin:.6rem 0;display:flex;justify-content:space-between;'
                'align-items:center;">'
                '<div>'
                '<div style="font-size:.62rem;color:#64748b;text-transform:uppercase;'
                'letter-spacing:.07em;margin-bottom:.12rem;">UPI ID</div>'
                '<div style="font-family:IBM Plex Mono,monospace;font-size:.87rem;'
                'font-weight:700;color:#1a56db;">' + str(UPI_ID) + '</div>'
                '</div>'
                '<div style="text-align:right;">'
                '<div style="font-size:.62rem;color:#64748b;text-transform:uppercase;'
                'letter-spacing:.07em;margin-bottom:.12rem;">Amount</div>'
                '<div style="font-size:1.1rem;font-weight:700;color:#0a1428;">'
                '&#8377;' + str(PLAN_PRICE) + '</div>'
                '</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;'
                'padding:.55rem .85rem;font-size:.75rem;color:#92400e;margin-bottom:.7rem;">'
                '&#9888; After paying, paste your 12-digit <strong>UTR / Transaction ID</strong>'
                ' from your UPI app below. Activation requires admin approval (within 24 h).</div>',
                unsafe_allow_html=True)
            _utr = st.text_input("UTR / Transaction ID", placeholder="e.g. 426781234567",
                                  key="pay_utr")
            st.markdown('<div class="ap">', unsafe_allow_html=True)
            if st.button("Submit for Verification", key="pay_act", use_container_width=True):
                _ok, _msg = _submit_payment(_uid_pay, _utr)
                if _ok:
                    st.success(_msg)
                    import time as _t; _t.sleep(1.5)
                    st.session_state.auth_page = "app"; st.rerun()
                else:
                    st.error(_msg)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ag">', unsafe_allow_html=True)
        if st.button("Back to App", key="pay_back", use_container_width=True):
            st.session_state.pop("rzp_order", None)
            st.session_state.auth_page = "app"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        _right_close()
    st.stop()
#  SIDEBAR  (only shown when logged in)
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="dm-logo">
      <div class="name">◈ DataMind AI</div>
      <div class="ver"><span class="dot"></span>Production · v5.0</div>
    </div>""", unsafe_allow_html=True)

    # ── User info panel ──────────────────────────────────
    _is_pending  = st.session_state.auth_plan == "pending_review"
    _plan_color  = ("#047857" if st.session_state.auth_plan == "premium"
                    else "#d97706" if _is_pending else "#b45309")
    _plan_label  = ("Premium" if st.session_state.auth_plan == "premium"
                    else "⏳ Pending" if _is_pending else "Free")
    _proj_left   = ("∞" if st.session_state.auth_plan == "premium"
                    else "—" if _is_pending
                    else str(max(0, PLAN_FREE_PROJ - st.session_state.auth_proj_used)))
    st.markdown(f"""
    <div style="background:#0f2a4a;border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem">
      <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:2px">Logged in as</div>
      <div style="font-size:0.82rem;color:#f1f5f9;font-weight:600;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
        {st.session_state.auth_email or "Guest"}</div>
      <div style="margin-top:6px;display:flex;gap:6px;align-items:center">
        <span style="background:{_plan_color};color:#fff;border-radius:20px;
          padding:1px 8px;font-size:0.65rem;font-weight:700">{_plan_label}</span>
        <span style="font-size:0.65rem;color:#94a3b8">Projects left: {_proj_left}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    if _is_pending:
        st.markdown(
            '<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:6px;'
            'padding:.5rem .75rem;font-size:.72rem;color:#92400e;margin-bottom:.5rem">'
            '⏳ Payment under review. You will be upgraded within 24 hours.</div>',
            unsafe_allow_html=True)
    elif st.session_state.auth_plan == "free":
        if st.button("⬆️ Upgrade ₹300/mo", use_container_width=True, key="sb_upgrade"):
            st.session_state.auth_page = "payment"
            st.rerun()

    if st.session_state.auth_plan == "premium" and st.session_state.auth_paid_until:
        _days = (st.session_state.auth_paid_until - datetime.datetime.utcnow()).days
        st.caption(f"Premium expires in **{max(0,_days)}** days")

    st.markdown("---")

    _nav_pages = [
        "📊 Analysis","🤖 AutoML","📈 Evaluation",
        "🔬 Rag","🔮 Inference","🔵 Clustering",
        "🧠 Deep Learning","💬 NLP / Text","🤖 Chatbot","🤖 Auto Labeling",
        "💳 Pricing",
    ]
    if st.session_state.auth_is_admin:
        _nav_pages.append("🛡️ Admin Panel")
    page = st.radio("Navigate", _nav_pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Training config</div>",unsafe_allow_html=True)
    test_size   = st.slider("Test split %",10,40,20)/100
    cv_folds    = st.slider("CV folds",3,10,5)
    random_seed = st.number_input("Random seed",value=42,step=1)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#64748b;line-height:2'>
    {'✓ Model loaded' if st.session_state.model else '— No model'}<br>
    Experiments: {len(st.session_state.history)}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    _sb1, _sb2 = st.columns(2)
    with _sb1:
        if st.button("🗑 Clear", use_container_width=True):
            for k in ["model","features","train_df","problem","label_encoders",
                      "y_test","preds","proba","col_meta","shap_values","shap_X","history",
                      "dl_model","dl_history","dl_type","dl_tokenizer","dl_classes","dl_img_size",
                      "nlp_model","nlp_vectorizer","nlp_classes",
                      "chatbot_model","chatbot_vectorizer","chatbot_responses",
                      "chatbot_classes","chat_history"]:
                st.session_state.pop(k, None)
            st.rerun()
    with _sb2:
        if st.button("🚪 Logout", use_container_width=True):
            _do_logout()
            st.rerun()

# ══════════════════════════════════════════════════════════
#  PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════
if page=="📊 Analysis":
    st.markdown("""<div class="dm-pagehead"><div class="icon">◈</div>
    <div><div class="title">Data Analysis</div>
    <div class="sub">Explore distributions, correlations and data quality</div></div></div>""",
    unsafe_allow_html=True)

    uploaded=st.file_uploader("Upload CSV dataset",type=["csv"])
    if not uploaded:
        st.markdown('<div class="dm-upload"><div class="uicon">⬆</div><div class="hint">Drop a CSV file to begin</div></div>',unsafe_allow_html=True)
        st.stop()

    fb=uploaded.read(); df_raw=load_csv(fb)

    c1,c2,c3,c4=st.columns(4)
    miss_total=int(df_raw.isnull().sum().sum())
    for col,val,lbl,cls in [
        (c1,f"{df_raw.shape[0]:,}","Rows",""),
        (c2,str(df_raw.shape[1]),"Columns","gold"),
        (c3,str(miss_total),"Missing values","red" if miss_total>0 else "green"),
        (c4,str(df_raw.select_dtypes(include=np.number).shape[1]),"Numeric cols","")]:
        col.markdown(f'<div class="dm-kpi {cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',unsafe_allow_html=True)

    t1,t2,t3,t4=st.tabs(["Overview","Distributions","Correlation","Missing"])

    with t1:
        st.markdown('<div class="dm-card"><div class="dm-card-title">Sample rows</div>',unsafe_allow_html=True)
        st.dataframe(df_raw.head(20),use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="dm-card"><div class="dm-card-title">Summary statistics</div>',unsafe_allow_html=True)
        st.dataframe(df_raw.describe().T.style.background_gradient(cmap="Blues"),use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with t2:
        num_cols=df_raw.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col_sel=st.selectbox("Select column",num_cols)
            fig,axes=plt.subplots(1,2,figsize=(10,3.5))
            axes[0].hist(df_raw[col_sel].dropna(),bins=30,color=C_BLUE,edgecolor="white",alpha=0.85,linewidth=0.5)
            axes[0].set_title(f"{col_sel} — Histogram")
            axes[1].boxplot(df_raw[col_sel].dropna(),patch_artist=True,
                            boxprops=dict(facecolor="#dbeafe",color=C_BLUE),
                            whiskerprops=dict(color=C_SLATE),capprops=dict(color=C_SLATE),
                            medianprops=dict(color=C_GOLD,linewidth=2))
            axes[1].set_title(f"{col_sel} — Boxplot")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with t3:
        @st.cache_data(show_spinner=False)
        def _corr(b): return load_csv(b).select_dtypes(include=np.number).corr()
        corr=_corr(fb)
        fig,ax=plt.subplots(figsize=(max(6,len(corr)*0.6),max(5,len(corr)*0.55)))
        sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),annot=True,fmt=".2f",
                    cmap="RdBu_r",center=0,linewidths=0.4,linecolor="#e2e6ea",ax=ax,
                    cbar_kws={"shrink":0.8},annot_kws={"size":8})
        ax.set_title("Feature Correlation Matrix"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with t4:
        miss=df_raw.isnull().sum().sort_values(ascending=False); miss=miss[miss>0]
        if miss.empty: st.success("✅ No missing values — dataset is complete.")
        else:
            fig,ax=plt.subplots(figsize=(8,max(3,len(miss)*0.42)))
            bars=ax.barh(miss.index,miss.values,color=C_RED,alpha=0.8,edgecolor="white")
            ax.bar_label(bars,padding=4,fontsize=8,color=C_SLATE)
            ax.set_xlabel("Missing count"); ax.set_title("Missing Values by Column")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════
#  PAGE: AUTOML
# ══════════════════════════════════════════════════════════
elif page=="🤖 AutoML":
    st.markdown("""<div class="dm-pagehead"><div class="icon">⚙</div>
    <div><div class="title">AutoML Training</div>
    <div class="sub">Configure, train and evaluate machine learning models</div></div></div>""",
    unsafe_allow_html=True)

    uploaded=st.file_uploader("Upload training CSV",type=["csv"])
    if not uploaded:
        st.markdown('<div class="dm-upload"><div class="uicon">⬆</div><div class="hint">Drop a CSV file to begin training</div></div>',unsafe_allow_html=True)
        st.stop()

    file_bytes=uploaded.read()
    df,encoders=cached_feature_engineering(file_bytes)
    st.session_state.label_encoders=encoders

    df_raw_meta=load_csv(file_bytes)
    st.session_state.col_meta={
        col:{"type":"cat","values":sorted(df_raw_meta[col].dropna().unique().tolist())}
             if df_raw_meta[col].dtype=="object" else {"type":"num","values":[]}
        for col in df_raw_meta.columns}

    st.markdown('<div class="dm-card"><div class="dm-card-title">Target & Feature selection</div>',unsafe_allow_html=True)
    c1,c2=st.columns([1,2])
    with c1: target=st.selectbox("🎯 Target column",df.columns)
    with c2:
        feat_cols=[c for c in df.columns if c!=target]
        features=st.multiselect("📐 Feature columns",feat_cols,default=feat_cols)
    st.markdown('</div>',unsafe_allow_html=True)

    if not features: st.warning("Select at least one feature."); st.stop()

    X,y=df[features],df[target]
    leaky=detect_leakage(X,y)
    if leaky:
        for col,corr in leaky: st.error(f"🚨 Leakage: **{col}** (corr={corr})")
        st.stop()

    is_cls=y.nunique()<=20 or y.dtype=="object"
    problem="Classification" if is_cls else "Regression"
    st.session_state.problem=problem
    st.markdown(f'<span class="dm-badge {"blue" if is_cls else "gold"}">{"🗂 " if is_cls else "📈 "}{problem}</span>',unsafe_allow_html=True)

    st.markdown('<div class="dm-card" style="margin-top:1rem"><div class="dm-card-title">Model configuration</div>',unsafe_allow_html=True)
    if problem=="Classification":
        mopts={"Random Forest":RandomForestClassifier(n_estimators=100,n_jobs=N_JOBS,random_state=random_seed,max_features="sqrt"),
               "Hist Gradient Boosting ⚡":HistGradientBoostingClassifier(max_iter=100,random_state=random_seed),
               "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=random_seed),
               "Logistic Regression":LogisticRegression(max_iter=500,n_jobs=N_JOBS,random_state=random_seed,solver="saga"),
               "KNN":KNeighborsClassifier(n_jobs=N_JOBS,algorithm="ball_tree"),
               "Decision Tree":DecisionTreeClassifier(random_state=random_seed)}
    else:
        mopts={"Random Forest":RandomForestRegressor(n_estimators=100,n_jobs=N_JOBS,random_state=random_seed,max_features="sqrt"),
               "Hist Gradient Boosting ⚡":HistGradientBoostingRegressor(max_iter=100,random_state=random_seed),
               "Gradient Boosting":GradientBoostingRegressor(n_estimators=100,random_state=random_seed),
               "Linear Regression":LinearRegression(n_jobs=N_JOBS),
               "Ridge":Ridge(),"Lasso":Lasso(),
               "Decision Tree":DecisionTreeRegressor(random_state=random_seed)}

    c1,c2,c3=st.columns(3)
    with c1: model_name=st.selectbox("Model",list(mopts.keys()))
    with c2: use_scaling=st.checkbox("Feature scaling",value=True)
    with c3: use_calib=st.checkbox("Probability calibration",value=is_cls)
    st.markdown('</div>',unsafe_allow_html=True)

    steps=([("scaler",StandardScaler())] if use_scaling else [])+[("model",mopts[model_name])]
    pipeline=Pipeline(steps)
    if use_calib and problem=="Classification":
        pipeline=CalibratedClassifierCV(pipeline,cv=3)

    if st.button("🚀 Train Model"):
        # ── Project limit gate ──────────────────────────────
        if st.session_state.auth_plan == "free" and st.session_state.auth_proj_used >= PLAN_FREE_PROJ:
            st.warning(
                f"You have used all {PLAN_FREE_PROJ} free projects. "
                "Upgrade to Premium (₹300/month) for unlimited projects.")
            if st.button("⬆️ Upgrade Now", key="automl_upgrade_btn"):
                st.session_state.auth_page = "payment"
                st.rerun()
            st.stop()
        _increment_project(st.session_state.auth_uid)
        # ───────────────────────────────────────────────────
        with st.spinner("Training in progress…"):
            strat=y if (problem=="Classification" and y.nunique()>1) else None
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_seed,stratify=strat)
            pipeline.fit(X_train,y_train); preds=pipeline.predict(X_test); proba=None

            if problem=="Classification":
                score=accuracy_score(y_test,preds); f1=f1_score(y_test,preds,average="weighted")
                try: proba=pipeline.predict_proba(X_test); auc=roc_auc_score(y_test,proba[:,1]) if y.nunique()==2 else None
                except: auc=None
            else:
                score=r2_score(y_test,preds); mae=mean_absolute_error(y_test,preds)
                rmse=mean_squared_error(y_test,preds)**0.5; f1=auc=None

            cv_scores=cross_val_score(pipeline,X,y,cv=cv_folds,n_jobs=N_JOBS,
                                      scoring="accuracy" if problem=="Classification" else "r2")

            # SHAP — premium only
            if st.session_state.auth_plan == "premium":
                sv=compute_shap(pipeline,X_train,X_test,features)
                st.session_state.shap_values=sv; st.session_state.shap_X=X_test
            else:
                st.session_state.shap_values=None; st.session_state.shap_X=None

        st.session_state.update(model=pipeline,features=features,train_df=X,
                                 y_test=y_test,preds=preds,proba=proba)

        record={"model":model_name,"problem":problem,"score":round(score,4),
                "cv_mean":round(cv_scores.mean(),4),"cv_std":round(cv_scores.std(),4),
                "features":features,"rows":int(X.shape[0])}
        st.session_state.history.append(record)

        best=max(st.session_state.history,key=lambda x:x["score"])

        kpis=([(f"Accuracy",f"{score:.3f}",""),("F1 Score",f"{f1:.3f}","gold"),
               ("CV Mean",f"{cv_scores.mean():.3f}","green"),("AUC-ROC",f"{auc:.3f}" if auc else "N/A","")]
              if problem=="Classification" else
              [("R² Score",f"{score:.3f}",""),("MAE",f"{mae:.3f}","gold"),
               ("RMSE",f"{rmse:.3f}","red"),("CV Mean",f"{cv_scores.mean():.3f}","green")])
        st.markdown('<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">'
                    +"".join(f'<div class="dm-kpi {cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
                             for lbl,val,cls in kpis)+'</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="dm-card" style="display:flex;align-items:center;gap:0.75rem">'
                    f'<span class="dm-badge green">🏆 Best</span>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.82rem">{best["model"]} — {best["score"]}</span>'
                    f'</div>',unsafe_allow_html=True)

        if problem=="Classification":
            with st.expander("📋 Classification report"):
                st.code(classification_report(y_test,preds),language="text")

        # Feature importance plot
        try:
            inner=pipeline.base_estimator if use_calib else pipeline
            fi_m=inner.named_steps.get("model",inner) if hasattr(inner,"named_steps") else inner
            if hasattr(fi_m,"feature_importances_"):
                imp_df=pd.DataFrame({"Feature":features,"Importance":fi_m.feature_importances_})\
                         .sort_values("Importance",ascending=True).tail(15)
                fig,ax=plt.subplots(figsize=(7,max(3,len(imp_df)*0.4)))
                colors=[C_RED if v<0.01 else C_BLUE for v in imp_df["Importance"]]
                bars=ax.barh(imp_df["Feature"],imp_df["Importance"],color=colors,alpha=0.82,edgecolor="white",linewidth=0.5)
                ax.bar_label(bars,fmt="%.4f",padding=4,fontsize=7.5,color=C_SLATE)
                ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                ax.set_title("Feature Importance (top 15)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except: pass

        st.download_button("💾 Download model (.pkl)",data=pickle.dumps(pipeline),
                           file_name=f"datamind_{model_name.lower().replace(' ','_')}.pkl",
                           mime="application/octet-stream")

    if st.session_state.history:
        st.markdown('<div class="dm-card"><div class="dm-card-title">Experiment history</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history),use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE: EVALUATION
# ══════════════════════════════════════════════════════════
elif page=="📈 Evaluation":
    st.markdown("""<div class="dm-pagehead"><div class="icon">📈</div>
    <div><div class="title">Model Evaluation</div>
    <div class="sub">Confusion matrix, ROC, calibration, fairness & residuals</div></div></div>""",
    unsafe_allow_html=True)

    if not st.session_state.model: st.info("Train a model first in the AutoML page."); st.stop()

    problem=st.session_state.problem; y_test=st.session_state.y_test
    preds=st.session_state.preds; proba=st.session_state.proba
    _yt=y_test.values.astype(np.float64).tobytes()
    _yp=preds.astype(np.float64).tobytes()

    if problem=="Classification":
        t1,t2,t3,t4=st.tabs(["Confusion Matrix","ROC Curve","Calibration","Fairness"])
        with t1: st.image(_plot_cm(_yt,_yp))
        with t2:
            if proba is not None and y_test.nunique()==2: st.image(_plot_roc(_yt,proba[:,1].astype(np.float64).tobytes()))
            else: st.info("Requires binary classification with probability support.")
        with t3:
            if proba is not None and y_test.nunique()==2: st.image(_plot_cal(_yt,proba[:,1].astype(np.float64).tobytes()))
            else: st.info("Requires binary classification with probability support.")
        with t4:
            fairness=fairness_check(y_test.values,preds)
            cols=st.columns(len(fairness))
            for col,(cls,acc) in zip(cols,fairness.items()):
                c="green" if acc>=0.8 else "gold" if acc>=0.6 else "red"
                col.markdown(f'<div class="dm-kpi {c}"><div class="val">{acc:.1%}</div><div class="lbl">Class {cls}</div></div>',unsafe_allow_html=True)
    else:
        t1,t2=st.tabs(["Residuals","Actual vs Predicted"])
        with t1: st.image(_plot_res(_yt,_yp))
        with t2: st.image(_plot_avp(_yt,_yp))

# ══════════════════════════════════════════════════════════
#  PAGRag
# ══════════════════════════════════════════════════════════
elif page=="🔬 Rag":
    st.markdown("""<div class="dm-pagehead"><div class="icon">🔬</div>
    <div><div class="title">RAG — Retrieval-Augmented Generation</div>
    <div class="sub">Upload documents · Name your project · Query with semantic search</div></div></div>""",
    unsafe_allow_html=True)

    # ── session state ──
    for _k, _v in [("rag_chunks",[]),("rag_sources",[]),("rag_vectorizer",None),
                   ("rag_matrix",None),("rag_built",False),("rag_project","My Project"),
                   ("rag_projects",{})]:
        if _k not in st.session_state: st.session_state[_k] = _v

    # ── helpers ──
    def _chunk_text(text, source, chunk_size=300, overlap=50):
        words = text.split()
        chunks, sources = [], []
        step = max(1, chunk_size - overlap)
        for i in range(0, max(1, len(words) - overlap), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk); sources.append(source)
        return chunks, sources

    def _extract_text(file_obj, filename):
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("txt", "md"):
            return file_obj.read().decode("utf-8", errors="ignore")
        elif ext == "csv":
            df_r = pd.read_csv(file_obj)
            return " ".join(df_r.astype(str).apply(lambda r: " ".join(r), axis=1).tolist())
        elif ext == "pdf":
            try:
                import pdfplumber
                parts = []
                with pdfplumber.open(file_obj) as pdf:
                    for pg in pdf.pages:
                        t = pg.extract_text()
                        if t: parts.append(t)
                return "\n".join(parts)
            except ImportError:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file_obj)
                    return "\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    return ""
        return ""

    # ════════════════════════════════════════════
    #  TOP — Project name bar
    # ════════════════════════════════════════════
    st.markdown('<div class="dm-card"><div class="dm-card-title">📁 Project</div>',
                unsafe_allow_html=True)
    pj1, pj2, pj3 = st.columns([3, 1, 1])
    with pj1:
        project_name = st.text_input("Project name", value=st.session_state.rag_project,
                                     placeholder="e.g. Legal Docs, Research Papers…",
                                     key="rag_proj_input", label_visibility="collapsed")
    with pj2:
        if st.button("💾 Save Project", key="rag_save_proj", use_container_width=True):
            if st.session_state.rag_built and project_name.strip():
                st.session_state.rag_projects[project_name] = {
                    "chunks":  st.session_state.rag_chunks,
                    "sources": st.session_state.rag_sources,
                    "vectorizer": st.session_state.rag_vectorizer,
                    "matrix":  st.session_state.rag_matrix,
                }
                st.session_state.rag_project = project_name
                st.success(f"✅ Saved as **{project_name}**")
            else:
                st.warning("Build a knowledge base first, then save.")
    with pj3:
        saved_names = list(st.session_state.rag_projects.keys())
        if saved_names:
            load_sel = st.selectbox("Load saved", ["— select —"] + saved_names,
                                    key="rag_load_sel", label_visibility="collapsed")
            if load_sel != "— select —":
                proj = st.session_state.rag_projects[load_sel]
                st.session_state.rag_chunks     = proj["chunks"]
                st.session_state.rag_sources    = proj["sources"]
                st.session_state.rag_vectorizer = proj["vectorizer"]
                st.session_state.rag_matrix     = proj["matrix"]
                st.session_state.rag_built      = True
                st.session_state.rag_project    = load_sel
                st.rerun()

    if st.session_state.rag_built:
        st.markdown(
            f'<span class="dm-badge green">🟢 Active: <strong>{st.session_state.rag_project}</strong> · '
            f'{len(st.session_state.rag_chunks)} chunks · '
            f'{len(set(st.session_state.rag_sources))} file(s)</span>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  TWO COLUMNS — Build  |  Query
    # ════════════════════════════════════════════
    rag_left, rag_right = st.columns([1, 1], gap="large")

    # ── LEFT: Build ──────────────────────────────
    with rag_left:
        st.markdown("### 📂 Build Knowledge Base")
        st.markdown('<div class="dm-card"><div class="dm-card-title">Upload Documents</div>',
                    unsafe_allow_html=True)

        rag_files = st.file_uploader(
            "Upload TXT, MD, CSV or PDF files",
            type=["txt","md","csv","pdf"],
            accept_multiple_files=True, key="rag_upload")

        rc1, rc2 = st.columns(2)
        with rc1: chunk_size    = st.slider("Chunk size (words)",  50, 600, 300, 50, key="rag_cs")
        with rc2: chunk_overlap = st.slider("Overlap (words)",      0, 100,  50, 10, key="rag_ov")
        st.markdown('</div>', unsafe_allow_html=True)

        if rag_files:
            st.markdown(f'<span class="dm-badge blue">📄 {len(rag_files)} file(s) ready</span>',
                        unsafe_allow_html=True)
            if st.button("🔨 Build Knowledge Base", key="rag_build", use_container_width=True):
                with st.spinner("Parsing and indexing…"):
                    all_chunks, all_sources, errors = [], [], []
                    for f in rag_files:
                        raw = _extract_text(f, f.name)
                        if not raw.strip(): errors.append(f.name); continue
                        ch, sr = _chunk_text(raw, f.name, chunk_size, chunk_overlap)
                        all_chunks.extend(ch); all_sources.extend(sr)
                    if errors:
                        st.warning(f"⚠️ No text from: {', '.join(errors)}. "
                                   "For PDFs: `pip install pdfplumber`")
                    if not all_chunks:
                        st.error("No text extracted. Please upload readable files.")
                    else:
                        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
                        _vec = _TV(max_features=15000, ngram_range=(1,2),
                                   strip_accents="unicode", sublinear_tf=True)
                        _mat = _vec.fit_transform(all_chunks)
                        st.session_state.rag_chunks     = all_chunks
                        st.session_state.rag_sources    = all_sources
                        st.session_state.rag_vectorizer = _vec
                        st.session_state.rag_matrix     = _mat
                        st.session_state.rag_built      = True
                        st.session_state.rag_project    = project_name or "Untitled Project"
                        st.success(f"✅ {len(all_chunks)} chunks indexed from {len(rag_files)} file(s)!")
                        st.rerun()

        if st.session_state.rag_built:
            st.markdown("<div class='dm-divider'></div>", unsafe_allow_html=True)
            unique_src = list(dict.fromkeys(st.session_state.rag_sources))
            ks1, ks2 = st.columns(2)
            ks1.markdown(f'<div class="dm-kpi green"><div class="val">'
                         f'{len(st.session_state.rag_chunks)}</div>'
                         f'<div class="lbl">Total Chunks</div></div>', unsafe_allow_html=True)
            ks2.markdown(f'<div class="dm-kpi gold"><div class="val">{len(unique_src)}</div>'
                         f'<div class="lbl">Documents</div></div>', unsafe_allow_html=True)

            st.markdown("**Indexed files:**")
            for s in unique_src:
                cnt = st.session_state.rag_sources.count(s)
                st.markdown(f'<span class="dm-badge grey">📄 {s} &nbsp;({cnt} chunks)</span> ',
                            unsafe_allow_html=True)

            if st.button("🗑 Clear KB", key="rag_clear", use_container_width=True):
                for _k in ["rag_chunks","rag_sources"]:
                    st.session_state[_k] = []
                st.session_state.rag_vectorizer = None
                st.session_state.rag_matrix     = None
                st.session_state.rag_built      = False
                st.rerun()
        else:
            st.markdown('<div class="dm-upload"><div class="uicon">📚</div>'
                        '<div class="hint">Upload files then click Build</div></div>',
                        unsafe_allow_html=True)

    # ── RIGHT: Query ─────────────────────────────
    with rag_right:
        st.markdown("### 🔍 Query")

        if not st.session_state.rag_built:
            st.info("Build a knowledge base first on the left.")
        else:
            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

            top_k     = st.slider("Top-K results",       1, 10,  3, key="rag_topk")
            min_score = st.slider("Min relevance score", 0.0, 1.0, 0.05, 0.01, key="rag_minscore")

            query = st.text_area("💬 Your question",
                                 placeholder="e.g. What is the main conclusion of this document?",
                                 height=90, key="rag_query")

            if st.button("🔎 Search", key="rag_search", use_container_width=True) and query.strip():
                with st.spinner("Retrieving…"):
                    try:
                        q_vec  = st.session_state.rag_vectorizer.transform([query])
                        scores = _cos_sim(q_vec, st.session_state.rag_matrix)[0]
                        ranked = np.argsort(scores)[::-1]
                        top_idx = [i for i in ranked if scores[i] >= min_score][:top_k]

                        if not top_idx:
                            st.warning("No chunks above the threshold. Lower the Min score.")
                        else:
                            st.markdown(f"**{len(top_idx)} result(s) found:**")
                            q_words = set(query.lower().split())
                            for rank, idx in enumerate(top_idx, 1):
                                chunk_text = st.session_state.rag_chunks[idx]
                                src        = st.session_state.rag_sources[idx]
                                score_val  = float(scores[idx])
                                badge_col  = "green" if score_val > 0.3 else "gold" if score_val > 0.1 else "grey"
                                highlighted = " ".join(
                                    f"**{w}**" if w.lower().strip(".,;:!?\"'") in q_words else w
                                    for w in chunk_text.split())
                                st.markdown(
                                    f'<div class="dm-card" style="border-left:3px solid var(--accent);'
                                    f'padding:0.9rem 1.1rem;margin-bottom:0.6rem">'
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'margin-bottom:0.4rem">'
                                    f'<span class="dm-badge {badge_col}">#{rank} · {score_val:.3f}</span>'
                                    f'<span class="dm-badge grey">📄 {src}</span></div></div>',
                                    unsafe_allow_html=True)
                                st.markdown(highlighted)
                                st.markdown("---")

                            # score bar chart
                            top20 = [(i, float(scores[i])) for i in ranked[:20]]
                            fig_s, ax_s = plt.subplots(figsize=(6, 2.5))
                            ax_s.bar(range(len(top20)),
                                     [s for _, s in top20],
                                     color=[C_BLUE if i in top_idx else "#e5e7eb" for i, _ in top20],
                                     edgecolor="white")
                            ax_s.set_xlabel("Chunk rank"); ax_s.set_ylabel("Cosine score")
                            ax_s.set_title("Retrieval scores (blue = returned)")
                            fig_s.tight_layout(); st.pyplot(fig_s); plt.close(fig_s)

                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

            st.markdown("<div class='dm-divider'></div>", unsafe_allow_html=True)
            with st.expander("🗂 Browse all chunks"):
                unique_src2 = list(dict.fromkeys(st.session_state.rag_sources))
                browse_src  = st.selectbox("Filter by file", ["All"] + unique_src2,
                                           key="rag_browse_src")
                filtered = [(i, c, s) for i,(c,s) in enumerate(
                    zip(st.session_state.rag_chunks, st.session_state.rag_sources))
                    if browse_src == "All" or s == browse_src]
                st.caption(f"{len(filtered)} chunk(s)")
                for i, chunk, src in filtered[:50]:
                    st.markdown(f'<span class="dm-badge grey">#{i} · {src}</span>',
                                unsafe_allow_html=True)
                    st.text(chunk[:250] + ("…" if len(chunk) > 250 else ""))
                    st.markdown("---")

# ══════════════════════════════════════════════════════════
#  PAGE: INFERENCE
# ══════════════════════════════════════════════════════════
elif page=="🔮 Inference":
    st.markdown("""<div class="dm-pagehead"><div class="icon">🔮</div>
    <div><div class="title">Inference</div>
    <div class="sub">Manual prediction & batch scoring with SHAP explanation</div></div></div>""",
    unsafe_allow_html=True)

    if not st.session_state.model: st.info("Train a model first in the AutoML page."); st.stop()

    problem=st.session_state.problem; features=st.session_state.features
    col_meta=st.session_state.col_meta; label_encoders=st.session_state.label_encoders

    st.markdown("### Manual prediction")
    st.markdown('<div class="dm-card"><div class="dm-card-title">Enter feature values</div>',unsafe_allow_html=True)
    cols=st.columns(min(4,len(features))); raw_inputs={}
    for i,col_name in enumerate(features):
        with cols[i%len(cols)]:
            meta=col_meta.get(col_name,{"type":"num","values":[]})
            if meta["type"]=="cat" and meta["values"]:
                raw_inputs[col_name]=st.selectbox(col_name,meta["values"],key=f"inf_{col_name}")
            else:
                raw_inputs[col_name]=st.number_input(col_name,value=0.0,key=f"inf_{col_name}")
    st.markdown('</div>',unsafe_allow_html=True)

    if st.button("🔮 Predict"):
        arr=encode_inputs(raw_inputs,features,label_encoders)
        pred=st.session_state.model.predict(arr)[0]
        st.markdown(f"""<div class="dm-card" style="text-align:center;padding:2rem;border-top:3px solid var(--accent)">
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);font-family:'IBM Plex Mono',monospace">Prediction</div>
        <div style="font-size:2.8rem;font-weight:700;color:var(--accent);margin:0.4rem 0;font-family:'IBM Plex Mono',monospace">{pred}</div>
        </div>""",unsafe_allow_html=True)

        if problem=="Classification":
            try:
                prob=st.session_state.model.predict_proba(arr)
                classes=st.session_state.model.classes_
                fig,ax=plt.subplots(figsize=(6,2.5))
                ax.barh([str(c) for c in classes],prob[0],
                        color=[C_BLUE if c==pred else "#e5e7eb" for c in classes],edgecolor="white")
                ax.set_xlim(0,1); ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                ax.set_title(f"Class probabilities — confidence {np.max(prob):.1%}")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            except: pass

        # SHAP waterfall for this prediction — premium only
        if st.session_state.auth_plan != "premium":
            st.markdown('<span class="dm-badge gold">🔒 SHAP Explainability — Premium only</span>',
                        unsafe_allow_html=True)
        elif st.session_state.shap_values is not None:
            sv=st.session_state.shap_values; ma=np.abs(sv).mean(axis=0)
            order=np.argsort(ma)[-10:]
            fig,ax=plt.subplots(figsize=(6,max(3,len(order)*0.38)))
            ax.barh([features[i] for i in order],[ma[i] for i in order],
                    color=C_BLUE,alpha=0.8,edgecolor="white")
            ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Top feature impact (this prediction type)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<div class='dm-divider'></div>",unsafe_allow_html=True)
    st.markdown("### Batch inference")
    batch_file=st.file_uploader("Upload CSV for batch prediction",type=["csv"])
    if batch_file:
        b=batch_file.read(); batch_raw=load_csv(b); batch_proc,_=cached_feature_engineering(b)
        missing=[f for f in features if f not in batch_proc.columns]
        if missing: st.error(f"Missing columns: {missing}")
        else:
            batch_preds=st.session_state.model.predict(batch_proc[features])
            batch_raw["prediction"]=batch_preds
            st.dataframe(batch_raw.head(50),use_container_width=True)
            st.download_button("⬇ Download predictions",data=batch_raw.to_csv(index=False).encode(),
                               file_name="predictions.csv",mime="text/csv")

    st.markdown("<div class='dm-divider'></div>",unsafe_allow_html=True)
    st.markdown("### Data drift monitor")
    drift_file=st.file_uploader("Upload new data for drift check",type=["csv"])
    if drift_file and st.session_state.train_df is not None:
        db2=drift_file.read(); new_proc,_=cached_feature_engineering(db2)
        drifted=detect_drift(st.session_state.train_df,new_proc)
        if drifted: st.warning(f"⚠️ Drift detected in: {', '.join(drifted)}")
        else: st.success("✅ No significant drift detected.")

# ══════════════════════════════════════════════════════════
#  PAGE: FIREBASE
# ══════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════
#  PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.markdown("""<div class="dm-pagehead"><div class="icon">🔵</div>
    <div><div class="title">Clustering</div>
    <div class="sub">K-Means · DBSCAN · Agglomerative · PCA / t-SNE visualisation</div></div></div>""",
    unsafe_allow_html=True)

    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    cl_file = st.file_uploader("Upload CSV dataset for clustering", type=["csv"], key="cl_up")
    if not cl_file:
        st.markdown('<div class="dm-upload"><div class="uicon">🔵</div>'
                    '<div class="hint">Upload a CSV to begin clustering</div></div>',
                    unsafe_allow_html=True)
        st.stop()

    try:
        cl_raw = pd.read_csv(io.BytesIO(cl_file.read()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()

    st.markdown(f'<span class="dm-badge blue">📄 {len(cl_raw)} rows · {len(cl_raw.columns)} cols</span>',
                unsafe_allow_html=True)
    st.dataframe(cl_raw.head(5), use_container_width=True)

    # ── Column selection ──────────────────────────────────
    num_cols = cl_raw.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found. Clustering requires numeric features."); st.stop()

    st.markdown("#### ⚙️ Configuration")
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        cl_features = st.multiselect("Feature columns", num_cols,
                                     default=num_cols[:min(len(num_cols), 6)], key="cl_feats")
    with cfg2:
        cl_algo = st.selectbox("Algorithm",
                               ["K-Means", "DBSCAN", "Agglomerative"], key="cl_algo")

    if not cl_features:
        st.warning("Select at least 2 feature columns."); st.stop()

    # ── Algorithm-specific params ─────────────────────────
    p1, p2, p3 = st.columns(3)
    if cl_algo == "K-Means":
        with p1: cl_k      = int(st.slider("K (clusters)", 2, 15, 3, key="cl_k"))
        with p2: cl_init   = st.selectbox("Init method", ["k-means++", "random"], key="cl_init")
        with p3: cl_maxiter= int(st.number_input("Max iter", 100, 1000, 300, step=100, key="cl_mi"))
    elif cl_algo == "DBSCAN":
        with p1: cl_eps    = float(st.number_input("eps",   0.1, 10.0, 0.5, step=0.1, key="cl_eps"))
        with p2: cl_minpts = int(st.number_input("min_samples", 2, 50, 5, key="cl_mp"))
        with p3: cl_metric = st.selectbox("Metric", ["euclidean","manhattan","cosine"], key="cl_met")
    else:  # Agglomerative
        with p1: cl_k      = int(st.slider("K (clusters)", 2, 15, 3, key="cl_k2"))
        with p2: cl_link   = st.selectbox("Linkage", ["ward","complete","average","single"], key="cl_lnk")
        with p3: cl_aff    = st.selectbox("Affinity", ["euclidean","manhattan","cosine"],
                                          key="cl_aff",
                                          disabled=(cl_link == "ward"))

    # ── Viz options ───────────────────────────────────────
    v1, v2 = st.columns(2)
    with v1: cl_viz   = st.selectbox("Visualisation", ["PCA (2D)", "PCA (3D — first 3 PCs)", "Feature pair"], key="cl_viz")
    with v2: cl_scale = st.checkbox("Standardize features (recommended)", value=True, key="cl_sc")

    if cl_viz == "Feature pair" and len(cl_features) >= 2:
        fp1, fp2 = st.columns(2)
        with fp1: cl_xfeat = st.selectbox("X axis", cl_features, key="cl_xf")
        with fp2: cl_yfeat = st.selectbox("Y axis", cl_features,
                                          index=min(1, len(cl_features)-1), key="cl_yf")

    if st.button("🚀 Run Clustering", key="cl_run", use_container_width=True):
        try:
            with st.spinner("Fitting cluster model…"):
                cl_work = cl_raw[cl_features].dropna().copy()
                X_cl    = cl_work.values.astype(np.float64)
                if cl_scale:
                    scaler_cl = _SS()
                    X_cl = scaler_cl.fit_transform(X_cl)

                # ── Fit ──
                if cl_algo == "K-Means":
                    mdl_cl = KMeans(n_clusters=cl_k, init=cl_init,
                                    max_iter=cl_maxiter, random_state=42, n_init=10)
                elif cl_algo == "DBSCAN":
                    aff_arg = cl_metric if cl_metric != "cosine" else "euclidean"
                    mdl_cl  = DBSCAN(eps=cl_eps, min_samples=cl_minpts, metric=cl_metric)
                else:
                    aff_use = "euclidean" if cl_link == "ward" else cl_aff
                    mdl_cl  = AgglomerativeClustering(n_clusters=cl_k,
                                                      linkage=cl_link, metric=aff_use)

                labels_cl = mdl_cl.fit_predict(X_cl)
                st.session_state.cluster_model  = mdl_cl
                st.session_state.cluster_labels = labels_cl
                cl_work["_cluster"] = labels_cl
                st.session_state.cluster_df = cl_work

            # ── Metrics ───────────────────────────────────
            n_found = len(set(labels_cl)) - (1 if -1 in labels_cl else 0)
            noise   = int(np.sum(labels_cl == -1))
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="dm-kpi blue" style="border-top-color:#1a56db">'
                        f'<div class="val">{n_found}</div>'
                        f'<div class="lbl">Clusters found</div></div>', unsafe_allow_html=True)
            if noise > 0:
                m2.markdown(f'<div class="dm-kpi red"><div class="val">{noise}</div>'
                            f'<div class="lbl">Noise points</div></div>', unsafe_allow_html=True)
            try:
                mask = labels_cl != -1
                if mask.sum() > 1 and n_found > 1:
                    sil = silhouette_score(X_cl[mask], labels_cl[mask])
                    dbi = davies_bouldin_score(X_cl[mask], labels_cl[mask])
                    chi = calinski_harabasz_score(X_cl[mask], labels_cl[mask])
                    m2.markdown(f'<div class="dm-kpi green"><div class="val">{sil:.3f}</div>'
                                f'<div class="lbl">Silhouette ↑</div></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div class="dm-kpi gold"><div class="val">{dbi:.3f}</div>'
                                f'<div class="lbl">Davies-Bouldin ↓</div></div>', unsafe_allow_html=True)
                    m4.markdown(f'<div class="dm-kpi"><div class="val">{chi:.0f}</div>'
                                f'<div class="lbl">Calinski-Harabasz ↑</div></div>', unsafe_allow_html=True)
            except Exception:
                pass

            # ── Cluster size chart ─────────────────────────
            unique_cl, counts_cl = np.unique(labels_cl, return_counts=True)
            palette_cl = [C_BLUE, C_GOLD, C_GREEN, C_RED, C_PURPLE, C_SLATE,
                          "#0891b2","#d97706","#16a34a","#dc2626","#7c3aed","#64748b",
                          "#0e7490","#b45309","#15803d"]
            fig_sz, ax_sz = plt.subplots(figsize=(max(4, n_found * 0.8), 3))
            bar_labels = [f"Noise" if c == -1 else f"C{c}" for c in unique_cl]
            bar_colors = ["#9ca3af" if c == -1 else palette_cl[int(c) % len(palette_cl)]
                          for c in unique_cl]
            bars_sz = ax_sz.bar(bar_labels, counts_cl, color=bar_colors, edgecolor="white")
            ax_sz.bar_label(bars_sz, padding=3, fontsize=9)
            ax_sz.set_title("Cluster sizes"); ax_sz.set_ylabel("Count")
            fig_sz.tight_layout(); st.pyplot(fig_sz); plt.close(fig_sz)

            # ── Visualisation ──────────────────────────────
            st.markdown("#### 📊 Cluster Visualisation")
            if cl_viz in ("PCA (2D)", "PCA (3D — first 3 PCs)"):
                n_comp = 3 if "3D" in cl_viz else 2
                n_comp = min(n_comp, X_cl.shape[1])
                pca_cl = PCA(n_components=n_comp, random_state=42)
                X_pca  = pca_cl.fit_transform(X_cl)
                var_exp = pca_cl.explained_variance_ratio_

                if n_comp == 2 or X_cl.shape[1] < 3:
                    fig_v, ax_v = plt.subplots(figsize=(7, 5))
                    for ci in unique_cl:
                        mask_v = labels_cl == ci
                        col_v  = "#9ca3af" if ci == -1 else palette_cl[int(ci) % len(palette_cl)]
                        lbl_v  = "Noise" if ci == -1 else f"Cluster {ci}"
                        ax_v.scatter(X_pca[mask_v, 0], X_pca[mask_v, 1],
                                     c=col_v, label=lbl_v, alpha=0.7, s=28, edgecolors="white", lw=0.3)
                    ax_v.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
                    ax_v.set_ylabel(f"PC2 ({var_exp[1]:.1%})" if n_comp > 1 else "PC2")
                    ax_v.set_title("PCA 2D — cluster view")
                    ax_v.legend(fontsize=8, framealpha=0.85)
                    fig_v.tight_layout(); st.pyplot(fig_v); plt.close(fig_v)
                else:
                    fig_v = plt.figure(figsize=(8, 6))
                    ax_v  = fig_v.add_subplot(111, projection="3d")
                    for ci in unique_cl:
                        mask_v = labels_cl == ci
                        col_v  = "#9ca3af" if ci == -1 else palette_cl[int(ci) % len(palette_cl)]
                        lbl_v  = "Noise" if ci == -1 else f"Cluster {ci}"
                        ax_v.scatter(X_pca[mask_v,0], X_pca[mask_v,1], X_pca[mask_v,2],
                                     c=col_v, label=lbl_v, alpha=0.7, s=20)
                    ax_v.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
                    ax_v.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
                    ax_v.set_zlabel(f"PC3 ({var_exp[2]:.1%})")
                    ax_v.set_title("PCA 3D — cluster view")
                    ax_v.legend(fontsize=7)
                    fig_v.tight_layout(); st.pyplot(fig_v); plt.close(fig_v)

                    # also show 2D projection
                    fig_v2, ax_v2 = plt.subplots(figsize=(7, 4))
                    for ci in unique_cl:
                        mask_v = labels_cl == ci
                        col_v  = "#9ca3af" if ci == -1 else palette_cl[int(ci) % len(palette_cl)]
                        ax_v2.scatter(X_pca[mask_v,0], X_pca[mask_v,1],
                                      c=col_v, alpha=0.7, s=20, edgecolors="white", lw=0.3)
                    ax_v2.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
                    ax_v2.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
                    ax_v2.set_title("PCA top-2 components")
                    fig_v2.tight_layout(); st.pyplot(fig_v2); plt.close(fig_v2)

            else:  # Feature pair
                xf = cl_xfeat if "cl_xfeat" in dir() else cl_features[0]
                yf = cl_yfeat if "cl_yfeat" in dir() else cl_features[min(1, len(cl_features)-1)]
                xi = cl_features.index(xf); yi = cl_features.index(yf)
                fig_v, ax_v = plt.subplots(figsize=(7, 5))
                for ci in unique_cl:
                    mask_v = labels_cl == ci
                    col_v  = "#9ca3af" if ci == -1 else palette_cl[int(ci) % len(palette_cl)]
                    lbl_v  = "Noise" if ci == -1 else f"Cluster {ci}"
                    ax_v.scatter(X_cl[mask_v, xi], X_cl[mask_v, yi],
                                 c=col_v, label=lbl_v, alpha=0.7, s=28, edgecolors="white", lw=0.3)
                ax_v.set_xlabel(xf); ax_v.set_ylabel(yf)
                ax_v.set_title(f"{xf} vs {yf} — by cluster")
                ax_v.legend(fontsize=8, framealpha=0.85)
                fig_v.tight_layout(); st.pyplot(fig_v); plt.close(fig_v)

            # ── K-Means elbow (only for K-Means) ──────────
            if cl_algo == "K-Means":
                with st.expander("📐 Elbow curve (inertia)"):
                    ks = list(range(2, min(12, len(cl_work))))
                    inertias = []
                    for ki in ks:
                        inertias.append(
                            KMeans(n_clusters=ki, random_state=42, n_init=5).fit(X_cl).inertia_)
                    fig_el, ax_el = plt.subplots(figsize=(6, 3))
                    ax_el.plot(ks, inertias, "o-", color=C_BLUE, lw=2, ms=6,
                               markerfacecolor="white", markeredgewidth=2)
                    ax_el.axvline(cl_k, color=C_GOLD, lw=1.5, linestyle="--",
                                  label=f"Current K={cl_k}")
                    ax_el.set_xlabel("K"); ax_el.set_ylabel("Inertia")
                    ax_el.set_title("Elbow curve — choose K at the bend")
                    ax_el.legend()
                    fig_el.tight_layout(); st.pyplot(fig_el); plt.close(fig_el)

            # ── Cluster profile table ──────────────────────
            st.markdown("#### 📋 Cluster Profiles (mean per feature)")
            prof_df = cl_work.groupby("_cluster")[cl_features].mean().round(3)
            prof_df.index = [f"Noise" if i == -1 else f"Cluster {i}" for i in prof_df.index]
            st.dataframe(prof_df.style.background_gradient(cmap="Blues", axis=0),
                         use_container_width=True)

            # ── Download labelled data ─────────────────────
            labelled_out = cl_raw.copy().iloc[cl_work.index]
            labelled_out["cluster"] = labels_cl
            st.download_button("⬇ Download labelled CSV",
                               data=labelled_out.to_csv(index=False).encode(),
                               file_name="clustered_data.csv", mime="text/csv")

        except Exception as exc:
            st.error(f"Clustering failed: {exc}")
            import traceback; st.code(traceback.format_exc(), language="text")

# ══════════════════════════════════════════════════════════
#  PAGE: DEEP LEARNING  (CNN images · LSTM text)
# ══════════════════════════════════════════════════════════
elif page == "🧠 Deep Learning":
    st.markdown("""<div class="dm-pagehead"><div class="icon">🧠</div>
    <div><div class="title">Deep Learning</div>
    <div class="sub">TensorFlow CNN for images · Bidirectional LSTM for text sequences</div></div></div>""",
    unsafe_allow_html=True)

    # ── Premium gate ──────────────────────────────────────
    if not _require_premium("Deep Learning (CNN / LSTM)"):
        st.stop()

    # ── TF availability gate ──────────────────────────────
    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow could not be imported.")
        if TF_IMPORT_ERR:
            st.code(TF_IMPORT_ERR, language="text")
        st.markdown("""
**Quick fixes based on your TF version:**
```bash
# TF 2.16+  (most common cause — Keras is now a separate package)
pip install tf-keras

# TF not found
pip install tensorflow

# Check what you have
python -c "import tensorflow as tf; print(tf.__version__)"
```""")
        st.stop()

    # ── Pillow check ──────────────────────────────────────
    try:
        from PIL import Image as _PILImage
        _PIL_OK = True
    except ImportError:
        _PIL_OK = False

    import base64 as _b64

    # ── Helper: safe feature-name getter ─────────────────
    def _feat_names(vec):
        try:
            return vec.get_feature_names_out().tolist()
        except AttributeError:
            return vec.get_feature_names()

    # ── Model builders (inside TF block) ─────────────────
    def _build_cnn(n_cls, h, w, filt, drop):
        _k = keras
        inp = _k.Input(shape=(h, w, 3))
        x   = layers.Rescaling(1.0 / 255.0)(inp)
        x   = layers.Conv2D(filt,    3, padding="same", activation="relu")(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.MaxPooling2D()(x)
        x   = layers.Conv2D(filt*2,  3, padding="same", activation="relu")(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.MaxPooling2D()(x)
        x   = layers.Conv2D(filt*4,  3, padding="same", activation="relu")(x)
        x   = layers.GlobalAveragePooling2D()(x)
        x   = layers.Dense(128, activation="relu")(x)
        x   = layers.Dropout(drop)(x)
        out = layers.Dense(n_cls, activation="softmax")(x)
        mdl = _k.Model(inp, out)
        mdl.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        return mdl

    def _build_lstm(n_cls, vocab, maxlen, emb, units):
        _k = keras
        inp = _k.Input(shape=(maxlen,))
        x   = layers.Embedding(vocab, emb, mask_zero=True)(inp)
        x   = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
        x   = layers.GlobalAveragePooling1D()(x)
        x   = layers.Dense(64, activation="relu")(x)
        x   = layers.Dropout(0.3)(x)
        out = layers.Dense(n_cls, activation="softmax")(x)
        mdl = _k.Model(inp, out)
        mdl.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        return mdl

    dl_tab1, dl_tab2 = st.tabs(["🖼 CNN — Image Classification", "📝 LSTM — Text Sequences"])

    # ══════════════════════════════════════════
    #  CNN TAB
    # ══════════════════════════════════════════
    with dl_tab1:
        st.markdown(
            '<div class="dm-card"><div class="dm-card-title">CNN Image Classifier</div>'
            '<p style="font-size:0.82rem;color:#374151;margin:0 0 0.5rem">'
            'Define your class labels (2 or more), then upload image files directly for each class. '
            'ImageDataGenerator augmentation is applied automatically during training.</p></div>',
            unsafe_allow_html=True)

        if not _PIL_OK:
            st.warning("⚠️ Pillow is not installed. Run: `pip install pillow`")

        ca1, ca2, ca3 = st.columns(3)
        with ca1: cnn_h   = int(st.number_input("Image height (px)", value=64, min_value=32, max_value=224, step=16, key="cnn_h"))
        with ca2: cnn_w   = int(st.number_input("Image width (px)",  value=64, min_value=32, max_value=224, step=16, key="cnn_w"))
        with ca3: cnn_ep  = int(st.slider("Epochs", 1, 30, 8, key="cnn_ep"))
        cb1, cb2 = st.columns(2)
        with cb1: cnn_filt = int(st.selectbox("Conv filters", [16, 32, 64], index=1, key="cnn_filt"))
        with cb2: cnn_drop = float(st.slider("Dropout", 0.0, 0.6, 0.3, step=0.1, key="cnn_drop"))

        # ── ImageDataGenerator augmentation options ──
        with st.expander("⚙️ ImageDataGenerator Augmentation Settings"):
            aug1, aug2, aug3 = st.columns(3)
            with aug1:
                aug_rot   = float(st.slider("Rotation range (°)",  0, 45, 15, key="aug_rot"))
                aug_zoom  = float(st.slider("Zoom range",          0.0, 0.4, 0.1, step=0.05, key="aug_zoom"))
            with aug2:
                aug_hflip = st.checkbox("Horizontal flip", value=True,  key="aug_hflip")
                aug_vflip = st.checkbox("Vertical flip",   value=False, key="aug_vflip")
            with aug3:
                aug_bright = float(st.slider("Brightness shift",  0.0, 0.5, 0.2, step=0.05, key="aug_bright"))
                aug_wshift = float(st.slider("Width shift range", 0.0, 0.3, 0.1, step=0.05, key="aug_wshift"))

        # ── Dynamic label management ──
        st.markdown("#### 🏷 Define Class Labels")
        if "cnn_labels" not in st.session_state:
            st.session_state.cnn_labels = ["class_0", "class_1"]

        lc_col, lc_btn = st.columns([3, 1])
        with lc_col:
            updated_labels = []
            lbl_cols = st.columns(min(len(st.session_state.cnn_labels), 4))
            for i, lbl in enumerate(st.session_state.cnn_labels):
                with lbl_cols[i % len(lbl_cols)]:
                    updated_labels.append(
                        st.text_input(f"Label {i+1}", value=lbl, key=f"cnn_lbl_{i}")
                    )
            st.session_state.cnn_labels = updated_labels
        with lc_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add label", key="cnn_add_lbl", use_container_width=True):
                st.session_state.cnn_labels.append(f"class_{len(st.session_state.cnn_labels)}")
                st.rerun()
            if len(st.session_state.cnn_labels) > 2:
                if st.button("➖ Remove last", key="cnn_rem_lbl", use_container_width=True):
                    st.session_state.cnn_labels.pop()
                    st.rerun()

        # ── Per-class image uploaders ──
        st.markdown("#### 📂 Upload Images per Class")
        cnn_class_files = {}
        n_labels = len(st.session_state.cnn_labels)
        cols_per_row = min(n_labels, 3)
        label_chunks = [st.session_state.cnn_labels[i:i+cols_per_row]
                        for i in range(0, n_labels, cols_per_row)]

        for chunk in label_chunks:
            row_cols = st.columns(len(chunk))
            for col, lbl in zip(row_cols, chunk):
                with col:
                    files = st.file_uploader(
                        f"📁 **{lbl}**", type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True, key=f"cnn_imgs_{lbl}"
                    )
                    cnn_class_files[lbl] = files or []
                    st.caption(f"{len(cnn_class_files[lbl])} image(s) uploaded")

        total_imgs = sum(len(v) for v in cnn_class_files.values())
        classes_with_imgs = {k: v for k, v in cnn_class_files.items() if len(v) > 0}

        if total_imgs > 0:
            if not _PIL_OK:
                st.error("Cannot process images — Pillow is required. Run: `pip install pillow`")
            else:
                n_cls_cnn = len(classes_with_imgs)
                palette = [C_BLUE, C_GOLD, C_GREEN, C_RED, C_PURPLE, C_SLATE]
                st.success(f"✅ {total_imgs} images across {n_cls_cnn} classes: "
                           f"{list(classes_with_imgs.keys())}")

                # Class distribution bar chart
                fig_dist, ax_dist = plt.subplots(figsize=(max(4, n_cls_cnn * 0.9), 2.8))
                ax_dist.bar(list(classes_with_imgs.keys()),
                            [len(v) for v in classes_with_imgs.values()],
                            color=[palette[i % len(palette)] for i in range(n_cls_cnn)],
                            edgecolor="white")
                ax_dist.set_title("Images per class"); ax_dist.set_ylabel("Count")
                fig_dist.tight_layout(); st.pyplot(fig_dist); plt.close(fig_dist)

                if n_cls_cnn < 2:
                    st.error("Need images in at least 2 classes to train.")
                elif st.button("🚀 Train CNN", key="cnn_train_btn"):
                    try:
                        with st.spinner("Loading images and training CNN with augmentation…"):
                            label_list = sorted(classes_with_imgs.keys())
                            lmap = {l: i for i, l in enumerate(label_list)}
                            imgs, lbls = [], []
                            bad = 0

                            for lbl, files in classes_with_imgs.items():
                                for f in files:
                                    try:
                                        pil_img = (_PILImage.open(f)
                                                   .convert("RGB")
                                                   .resize((cnn_w, cnn_h)))
                                        imgs.append(np.array(pil_img, dtype=np.float32))
                                        lbls.append(lmap[lbl])
                                    except Exception:
                                        bad += 1

                            if bad > 0:
                                st.warning(f"⚠️ Skipped {bad} unreadable image(s).")
                            if len(imgs) < 4:
                                st.error("Not enough valid images (need at least 4).")
                                st.stop()

                            X_cnn = np.stack(imgs)
                            y_cnn = np.array(lbls, dtype=np.int32)
                            perm  = np.random.permutation(len(X_cnn))
                            sp    = max(1, int(0.8 * len(perm)))
                            Xtr, Xte = X_cnn[perm[:sp]], X_cnn[perm[sp:]]
                            ytr, yte = y_cnn[perm[:sp]], y_cnn[perm[sp:]]

                            # ── ImageDataGenerator ──
                            try:
                                from tensorflow.keras.preprocessing.image import ImageDataGenerator as _IDG
                            except ImportError:
                                try:
                                    from keras.preprocessing.image import ImageDataGenerator as _IDG
                                except ImportError:
                                    from tf_keras.preprocessing.image import ImageDataGenerator as _IDG

                            bright_rng = ([1.0 - aug_bright, 1.0 + aug_bright]
                                          if aug_bright > 0 else None)
                            datagen = _IDG(
                                rotation_range=aug_rot,
                                width_shift_range=aug_wshift,
                                zoom_range=aug_zoom,
                                horizontal_flip=aug_hflip,
                                vertical_flip=aug_vflip,
                                brightness_range=bright_rng,
                            )
                            datagen.fit(Xtr)

                            cnn_mdl = _build_cnn(len(label_list), cnn_h, cnn_w,
                                                  cnn_filt, cnn_drop)
                            es_cnn  = keras.callbacks.EarlyStopping(
                                patience=3, restore_best_weights=True,
                                monitor="val_accuracy")
                            batch_sz = min(32, max(4, len(Xtr) // 4))
                            hist_c = cnn_mdl.fit(
                                datagen.flow(Xtr, ytr, batch_size=batch_sz),
                                steps_per_epoch=max(1, len(Xtr) // batch_sz),
                                epochs=cnn_ep,
                                validation_data=(Xte, yte),
                                callbacks=[es_cnn], verbose=0
                            )
                            st.session_state.dl_model    = cnn_mdl
                            st.session_state.dl_history  = hist_c.history
                            st.session_state.dl_type     = "cnn"
                            st.session_state.dl_classes  = label_list
                            st.session_state.dl_img_size = (cnn_h, cnn_w)

                        h_c = hist_c.history
                        val_acc_c  = float(max(h_c.get("val_accuracy", [0])))
                        val_loss_c = float(min(h_c.get("val_loss",     [0])))
                        k1, k2, k3 = st.columns(3)
                        k1.markdown(f'<div class="dm-kpi green"><div class="val">{val_acc_c:.3f}</div><div class="lbl">Val Accuracy</div></div>', unsafe_allow_html=True)
                        k2.markdown(f'<div class="dm-kpi"><div class="val">{val_loss_c:.3f}</div><div class="lbl">Val Loss</div></div>', unsafe_allow_html=True)
                        k3.markdown(f'<div class="dm-kpi gold"><div class="val">{len(label_list)}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)

                        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                        axes[0].plot(h_c.get("accuracy",     []), color=C_BLUE, lw=2, label="Train")
                        axes[0].plot(h_c.get("val_accuracy", []), color=C_GOLD, lw=2,
                                     linestyle="--", label="Val")
                        axes[0].set_title("Accuracy"); axes[0].legend()
                        axes[1].plot(h_c.get("loss",         []), color=C_BLUE, lw=2, label="Train")
                        axes[1].plot(h_c.get("val_loss",     []), color=C_GOLD, lw=2,
                                     linestyle="--", label="Val")
                        axes[1].set_title("Loss"); axes[1].legend()
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        with st.expander("📋 Model summary"):
                            lines_s = []
                            cnn_mdl.summary(print_fn=lambda x: lines_s.append(x))
                            st.code("\n".join(lines_s), language="text")

                    except Exception as exc:
                        st.error(f"Training failed: {exc}")
        else:
            st.markdown('<div class="dm-upload"><div class="uicon">🖼</div>'
                        '<div class="hint">Define your class labels above, then upload images for each class</div></div>',
                        unsafe_allow_html=True)

        # CNN inference
        if st.session_state.get("dl_type") == "cnn" and st.session_state.dl_model and _PIL_OK:
            st.markdown("---")
            st.markdown("### 🔮 Predict on new image")
            pred_img_f = st.file_uploader("Upload image to classify", type=["png","jpg","jpeg"], key="cnn_pred_f")
            if pred_img_f:
                try:
                    h2, w2  = st.session_state.dl_img_size
                    pil_in  = _PILImage.open(pred_img_f).convert("RGB").resize((w2, h2))
                    arr_in  = np.expand_dims(np.array(pil_in, dtype=np.float32), axis=0)
                    prob_c  = st.session_state.dl_model.predict(arr_in, verbose=0)[0]
                    cls_c   = st.session_state.dl_classes
                    pred_lc = cls_c[int(np.argmax(prob_c))]
                    ca, cb  = st.columns([1, 2])
                    with ca:
                        st.image(pil_in, caption="Input image", use_container_width=True)
                    with cb:
                        st.markdown(
                            f'<div class="dm-card" style="text-align:center;padding:1.5rem;border-top:3px solid var(--accent)">'
                            f'<div style="font-size:0.7rem;text-transform:uppercase;color:var(--muted)">Predicted class</div>'
                            f'<div style="font-size:2rem;font-weight:700;color:var(--accent);font-family:IBM Plex Mono,monospace">{pred_lc}</div>'
                            f'<div style="font-size:0.75rem;color:var(--muted)">Confidence: {float(np.max(prob_c)):.1%}</div>'
                            f'</div>', unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(5, max(2.5, len(cls_c)*0.4)))
                        ax.barh(cls_c, prob_c.tolist(),
                                color=[C_BLUE if c == pred_lc else "#e5e7eb" for c in cls_c],
                                edgecolor="white")
                        ax.set_xlim(0, 1)
                        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                        ax.set_title("Class probabilities")
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")

    # ══════════════════════════════════════════
    #  LSTM TAB
    # ══════════════════════════════════════════
    with dl_tab2:
        st.markdown(
            '<div class="dm-card"><div class="dm-card-title">LSTM Text Sequence Classifier</div>'
            '<p style="font-size:0.82rem;color:#374151;margin:0 0 0.5rem">'
            'Upload <strong>any CSV</strong>. Select your text column and — optionally — a label column. '
            'If no label is selected, text statistics are shown only. '
            'A label column is required to train the LSTM classifier.</p></div>',
            unsafe_allow_html=True)

        la1, la2, la3 = st.columns(3)
        with la1: lstm_ml  = int(st.number_input("Max sequence length", value=100, min_value=10, max_value=512, step=10, key="lstm_ml"))
        with la2: lstm_voc = int(st.number_input("Vocab size",          value=10000, min_value=500, max_value=50000, step=500, key="lstm_voc"))
        with la3: lstm_ep  = int(st.slider("Epochs", 1, 30, 8, key="lstm_ep"))
        lb1, lb2 = st.columns(2)
        with lb1: lstm_emb   = int(st.selectbox("Embedding dim", [32, 64, 128], index=1, key="lstm_emb"))
        with lb2: lstm_units = int(st.selectbox("LSTM units",    [32, 64, 128], index=1, key="lstm_un"))

        lstm_file = st.file_uploader("Upload any CSV", type=["csv"], key="lstm_up")

        if lstm_file:
            try:
                lstm_raw = pd.read_csv(io.BytesIO(lstm_file.read()))
                st.success(f"✅ {len(lstm_raw)} rows · {len(lstm_raw.columns)} columns loaded")
                st.dataframe(lstm_raw.head(5), use_container_width=True)

                # ── Column selection ──
                all_lstm_cols = list(lstm_raw.columns)
                str_lstm_cols = ([c for c in all_lstm_cols if lstm_raw[c].dtype == object]
                                 or all_lstm_cols)
                lsc1, lsc2 = st.columns(2)
                with lsc1:
                    lstm_tcol = st.selectbox("📄 Text column", str_lstm_cols, key="lstm_tcol")
                with lsc2:
                    lstm_lcol = st.selectbox(
                        "🏷 Label column (optional — select '— none —' to skip)",
                        ["— none —"] + all_lstm_cols, key="lstm_lcol"
                    )

                has_label = lstm_lcol != "— none —"

                # ── Text stats (always shown) ──
                texts_preview = lstm_raw[lstm_tcol].dropna().astype(str).str.strip()
                texts_preview = texts_preview[texts_preview.str.len() > 0]
                word_lengths  = texts_preview.str.split().str.len()
                sp1, sp2, sp3 = st.columns(3)
                sp1.markdown(f'<div class="dm-kpi"><div class="val">{len(texts_preview)}</div><div class="lbl">Texts</div></div>', unsafe_allow_html=True)
                sp2.markdown(f'<div class="dm-kpi gold"><div class="val">{int(word_lengths.mean()) if len(word_lengths) > 0 else 0}</div><div class="lbl">Avg Words</div></div>', unsafe_allow_html=True)
                sp3.markdown(f'<div class="dm-kpi"><div class="val">{int(word_lengths.max()) if len(word_lengths) > 0 else 0}</div><div class="lbl">Max Words</div></div>', unsafe_allow_html=True)

                if not has_label:
                    st.info("ℹ️ No label column selected — showing text statistics only. "
                            "Select a label column above to enable LSTM training.")
                else:
                    # ── Build working dataframe ──
                    lstm_work = lstm_raw[[lstm_tcol, lstm_lcol]].copy()
                    lstm_work.columns = ["text", "label"]
                    lstm_work["text"] = lstm_work["text"].astype(str).str.strip()
                    lstm_work = lstm_work.dropna(subset=["text", "label"])
                    lstm_work = lstm_work[lstm_work["text"].str.len() > 0]

                    n_cls_l = lstm_work["label"].nunique()
                    cls_names = sorted(lstm_work["label"].astype(str).unique().tolist())
                    st.success(f"✅ {len(lstm_work)} usable rows · {n_cls_l} classes: {cls_names}")

                    if n_cls_l < 2:
                        st.error("Need at least 2 unique labels to train.")
                    elif len(lstm_work) < 10:
                        st.error("Need at least 10 rows to train.")
                    elif st.button("🚀 Train LSTM", key="lstm_train_btn"):
                        try:
                            with st.spinner("Tokenizing text and training LSTM…"):
                                cls_l  = sorted(lstm_work["label"].astype(str).unique().tolist())
                                lmap_l = {l: i for i, l in enumerate(cls_l)}
                                y_l    = lstm_work["label"].astype(str).map(lmap_l).values.astype(np.int32)

                                tok_l  = KerasTokenizer(num_words=lstm_voc, oov_token="<OOV>")
                                tok_l.fit_on_texts(lstm_work["text"].tolist())
                                seqs_l = tok_l.texts_to_sequences(lstm_work["text"].tolist())
                                X_l    = pad_sequences(seqs_l, maxlen=lstm_ml, padding="post", truncating="post")

                                perm_l = np.random.permutation(len(X_l))
                                sp_l   = max(1, int(0.8 * len(perm_l)))
                                Xtr_l, Xte_l = X_l[perm_l[:sp_l]], X_l[perm_l[sp_l:]]
                                ytr_l, yte_l = y_l[perm_l[:sp_l]], y_l[perm_l[sp_l:]]

                                lstm_mdl = _build_lstm(len(cls_l), lstm_voc, lstm_ml, lstm_emb, lstm_units)
                                es_l     = keras.callbacks.EarlyStopping(
                                    patience=3, restore_best_weights=True, monitor="val_accuracy")
                                hist_l   = lstm_mdl.fit(
                                    Xtr_l, ytr_l, epochs=lstm_ep, batch_size=32,
                                    validation_data=(Xte_l, yte_l),
                                    callbacks=[es_l], verbose=0
                                )
                                st.session_state.dl_model     = lstm_mdl
                                st.session_state.dl_history   = hist_l.history
                                st.session_state.dl_type      = "lstm"
                                st.session_state.dl_tokenizer = tok_l
                                st.session_state.dl_classes   = cls_l
                                st.session_state["dl_maxlen"] = lstm_ml

                            h_l  = hist_l.history
                            va_l = float(max(h_l.get("val_accuracy", [0])))
                            k1l, k2l = st.columns(2)
                            k1l.markdown(f'<div class="dm-kpi green"><div class="val">{va_l:.3f}</div><div class="lbl">Val Accuracy</div></div>', unsafe_allow_html=True)
                            k2l.markdown(f'<div class="dm-kpi gold"><div class="val">{len(cls_l)}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)

                            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                            axes[0].plot(h_l.get("accuracy",     []), color=C_BLUE, lw=2, label="Train")
                            axes[0].plot(h_l.get("val_accuracy", []), color=C_GOLD, lw=2,
                                         linestyle="--", label="Val")
                            axes[0].set_title("Accuracy"); axes[0].legend()
                            axes[1].plot(h_l.get("loss",         []), color=C_BLUE, lw=2, label="Train")
                            axes[1].plot(h_l.get("val_loss",     []), color=C_GOLD, lw=2,
                                         linestyle="--", label="Val")
                            axes[1].set_title("Loss"); axes[1].legend()
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        except Exception as exc:
                            st.error(f"LSTM training failed: {exc}")

            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
        else:
            st.markdown('<div class="dm-upload"><div class="uicon">📝</div>'
                        '<div class="hint">Upload any CSV — select your text column and optionally a label column</div></div>',
                        unsafe_allow_html=True)

        # LSTM inference
        if st.session_state.get("dl_type") == "lstm" and st.session_state.dl_model:
            st.markdown("---")
            st.markdown("### 🔮 Classify new text")
            infer_txt = st.text_area("Enter text to classify", height=80, key="lstm_infer")
            if st.button("Classify", key="lstm_clf_btn") and infer_txt.strip():
                try:
                    ml2      = st.session_state.get("dl_maxlen", 100)
                    tok2     = st.session_state.dl_tokenizer
                    seq2     = tok2.texts_to_sequences([infer_txt])
                    pad2     = pad_sequences(seq2, maxlen=ml2, padding="post", truncating="post")
                    prob2    = st.session_state.dl_model.predict(pad2, verbose=0)[0]
                    cls2     = st.session_state.dl_classes
                    pred2    = cls2[int(np.argmax(prob2))]
                    st.markdown(
                        f'<div class="dm-card" style="text-align:center;padding:1.5rem;border-top:3px solid var(--accent)">'
                        f'<div style="font-size:0.7rem;text-transform:uppercase;color:var(--muted)">Predicted class</div>'
                        f'<div style="font-size:2rem;font-weight:700;color:var(--accent);font-family:IBM Plex Mono,monospace">{pred2}</div>'
                        f'<div style="font-size:0.75rem;color:var(--muted)">Confidence: {float(np.max(prob2)):.1%}</div>'
                        f'</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, max(2.5, len(cls2)*0.4)))
                    ax.barh(cls2, prob2.tolist(),
                            color=[C_BLUE if c == pred2 else "#e5e7eb" for c in cls2],
                            edgecolor="white")
                    ax.set_xlim(0, 1)
                    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                    ax.set_title("Class probabilities")
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


# ══════════════════════════════════════════════════════════
#  PAGE: NLP / TEXT
# ══════════════════════════════════════════════════════════
elif page == "💬 NLP / Text":
    st.markdown("""<div class="dm-pagehead"><div class="icon">💬</div>
    <div><div class="title">NLP / Text Classification</div>
    <div class="sub">TF-IDF · Word frequency · Sklearn text classifiers · Live inference</div></div></div>""",
    unsafe_allow_html=True)

    # ── Premium gate ──────────────────────────────────────
    if not _require_premium("NLP Text Classification"):
        st.stop()

    # Helper: safe feature names (sklearn < 1.0 vs >= 1.0)
    def _fn(vec):
        try:    return vec.get_feature_names_out().tolist()
        except: return vec.get_feature_names()

    nlp_file = st.file_uploader("Upload text CSV", type=["csv"], key="nlp_up")
    if not nlp_file:
        st.markdown('<div class="dm-upload"><div class="uicon">💬</div>'
                    '<div class="hint">Upload a CSV with at least one text column (and optionally a label column)</div></div>',
                    unsafe_allow_html=True)
        st.stop()

    try:
        nlp_raw = pd.read_csv(io.BytesIO(nlp_file.read()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()

    if nlp_raw.empty or len(nlp_raw.columns) == 0:
        st.error("CSV appears to be empty."); st.stop()

    # Detect text columns
    nlp_str_cols = [c for c in nlp_raw.columns if nlp_raw[c].dtype == object]
    if not nlp_str_cols:
        nlp_str_cols = list(nlp_raw.columns)   # fallback: all columns

    nc1, nc2 = st.columns(2)
    with nc1: nlp_tcol = st.selectbox("📄 Text column",             nlp_str_cols)
    with nc2: nlp_lcol = st.selectbox("🏷 Label column (optional)", ["— none —"] + list(nlp_raw.columns))

    nlp_texts = nlp_raw[nlp_tcol].dropna().astype(str).tolist()
    if len(nlp_texts) == 0:
        st.error("Selected text column is empty after dropping nulls."); st.stop()

    t_stat, t_wf, t_clf = st.tabs(["📊 Text Statistics", "🔤 Word Frequency", "🤖 Text Classifier"])

    # ── Statistics tab ────────────────────────────────────
    with t_stat:
        try:
            wlens = [len(t.split()) for t in nlp_texts]
            clens = [len(t)          for t in nlp_texts]
            s1, s2, s3, s4 = st.columns(4)
            s1.markdown(f'<div class="dm-kpi"><div class="val">{len(nlp_texts):,}</div><div class="lbl">Documents</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="dm-kpi gold"><div class="val">{int(np.mean(wlens))}</div><div class="lbl">Avg words</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="dm-kpi"><div class="val">{max(wlens)}</div><div class="lbl">Max words</div></div>', unsafe_allow_html=True)
            s4.markdown(f'<div class="dm-kpi green"><div class="val">{int(np.median(wlens))}</div><div class="lbl">Median words</div></div>', unsafe_allow_html=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].hist(wlens, bins=30, color=C_BLUE,   edgecolor="white", alpha=0.82)
            axes[0].set_xlabel("Word count"); axes[0].set_title("Word count distribution")
            axes[1].hist(clens, bins=30, color=C_PURPLE, edgecolor="white", alpha=0.82)
            axes[1].set_xlabel("Character count"); axes[1].set_title("Character count distribution")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.dataframe(nlp_raw.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Statistics error: {e}")

    # ── Word frequency tab ────────────────────────────────
    with t_wf:
        try:
            ng_sel   = st.radio("N-gram", ["Unigram", "Bigram", "Trigram"], horizontal=True)
            ng_rng   = {"Unigram": (1,1), "Bigram": (1,2), "Trigram": (1,3)}[ng_sel]
            top_n_wf = st.slider("Top N terms", 10, 50, 20, key="topn_wf")
            sw_raw   = st.text_input(
                "Stopwords (comma-separated, leave blank to disable)",
                value="the,a,an,is,in,on,at,to,of,and,or,it,this,that,for,with,as,are,was,be,has,had,have,he,she,they,we,i,you"
            )
            sw_list  = [w.strip() for w in sw_raw.split(",") if w.strip()] or None

            wf_vec  = TfidfVectorizer(ngram_range=ng_rng, stop_words=sw_list,
                                      max_features=top_n_wf * 5, sublinear_tf=True)
            wf_mat  = wf_vec.fit_transform(nlp_texts)
            wf_sum  = np.asarray(wf_mat.sum(axis=0)).ravel()
            wf_feat = _fn(wf_vec)
            tidx    = np.argsort(wf_sum)[-top_n_wf:]
            tterms  = [wf_feat[i] for i in tidx]
            tscores = [float(wf_sum[i]) for i in tidx]

            fig, ax = plt.subplots(figsize=(9, max(4, len(tterms)*0.38)))
            grad    = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(tterms)))
            bars    = ax.barh(tterms, tscores, color=grad, edgecolor="white", linewidth=0.4)
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=7.5, color=C_SLATE)
            ax.set_xlabel("TF-IDF score"); ax.set_title(f"Top {top_n_wf} {ng_sel}s by TF-IDF weight")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.error(f"Word frequency error: {e}")

    # ── Classifier tab ────────────────────────────────────
    with t_clf:
        if nlp_lcol == "— none —":
            st.info("Select a label column above to enable text classification training.")
        else:
            try:
                nlp_valid = nlp_raw[[nlp_tcol, nlp_lcol]].dropna().copy()
                nlp_valid[nlp_tcol] = nlp_valid[nlp_tcol].astype(str)
                nlp_valid[nlp_lcol] = nlp_valid[nlp_lcol].astype(str)

                if len(nlp_valid) < 4:
                    st.warning(f"Only {len(nlp_valid)} valid rows — need at least 4.")
                elif nlp_valid[nlp_lcol].nunique() < 2:
                    st.warning("Label column needs at least 2 unique classes.")
                else:
                    st.success(f"✅ {len(nlp_valid)} rows · {nlp_valid[nlp_lcol].nunique()} classes")

                    nc3a, nc3b, nc3c = st.columns(3)
                    with nc3a: nlp_clf_name = st.selectbox("Model", ["Logistic Regression", "LinearSVC", "Naive Bayes", "Random Forest"], key="nlp_clf")
                    with nc3b: nlp_mf       = int(st.number_input("TF-IDF max features", value=5000, min_value=500, max_value=50000, step=500, key="nlp_mf"))
                    with nc3c: nlp_ng_s     = st.selectbox("N-gram range", ["(1,1)", "(1,2)", "(1,3)"], key="nlp_ng")
                    nlp_ng_p = tuple(int(x) for x in nlp_ng_s.strip("()").split(","))

                    if st.button("🚀 Train Text Classifier", key="nlp_train_btn"):
                        try:
                            with st.spinner("Vectorizing and training…"):
                                txts_n = nlp_valid[nlp_tcol].tolist()
                                lbls_n = nlp_valid[nlp_lcol].tolist()
                                le_n   = LabelEncoder()
                                y_n    = le_n.fit_transform(lbls_n)

                                vec_n  = TfidfVectorizer(max_features=nlp_mf, ngram_range=nlp_ng_p,
                                                         sublinear_tf=True, strip_accents="unicode")
                                X_n    = vec_n.fit_transform(txts_n)

                                clf_opts = {
                                    "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, solver="saga", n_jobs=N_JOBS),
                                    "LinearSVC":           LinearSVC(max_iter=1000),
                                    "Naive Bayes":         MultinomialNB(),
                                    "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=42),
                                }
                                clf_n  = clf_opts[nlp_clf_name]

                                min_cls_n = int(nlp_valid[nlp_lcol].value_counts().min())
                                strat_n   = y_n if min_cls_n >= 2 else None
                                Xtr_n, Xte_n, ytr_n, yte_n = train_test_split(
                                    X_n, y_n, test_size=0.2, random_state=42, stratify=strat_n)
                                clf_n.fit(Xtr_n, ytr_n)
                                pn  = clf_n.predict(Xte_n)
                                an  = float(accuracy_score(yte_n, pn))
                                f1n = float(f1_score(yte_n, pn, average="weighted"))

                                st.session_state.nlp_model      = clf_n
                                st.session_state.nlp_vectorizer = vec_n
                                st.session_state.nlp_classes    = le_n.classes_.tolist()

                            nk1, nk2 = st.columns(2)
                            nk1.markdown(f'<div class="dm-kpi green"><div class="val">{an:.3f}</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
                            nk2.markdown(f'<div class="dm-kpi gold"><div class="val">{f1n:.3f}</div><div class="lbl">F1 Score (weighted)</div></div>', unsafe_allow_html=True)

                            with st.expander("📋 Full classification report"):
                                st.code(classification_report(yte_n, pn, target_names=le_n.classes_), language="text")

                            cm_n = confusion_matrix(yte_n, pn)
                            fig, ax = plt.subplots(figsize=(max(5, len(le_n.classes_)*0.9),
                                                             max(4, len(le_n.classes_)*0.8)))
                            sns.heatmap(cm_n, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                                        xticklabels=le_n.classes_, yticklabels=le_n.classes_, ax=ax)
                            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        except Exception as e:
                            st.error(f"Training error: {e}")

                    # Live inference (available after training)
                    if st.session_state.nlp_model is not None:
                        st.markdown("---")
                        st.markdown("### 🔮 Classify new text")
                        live_t = st.text_area("Type text to classify", height=80, key="nlp_live_t")
                        if st.button("Classify", key="nlp_live_btn") and live_t.strip():
                            try:
                                xnew  = st.session_state.nlp_vectorizer.transform([live_t])
                                pnew  = int(st.session_state.nlp_model.predict(xnew)[0])
                                lnew  = st.session_state.nlp_classes[pnew]
                                st.markdown(
                                    f'<div class="dm-card" style="text-align:center;padding:1.5rem;border-top:3px solid var(--accent)">'
                                    f'<div style="font-size:0.7rem;text-transform:uppercase;color:var(--muted)">Predicted class</div>'
                                    f'<div style="font-size:2rem;font-weight:700;color:var(--accent);font-family:IBM Plex Mono,monospace">{lnew}</div>'
                                    f'</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Inference error: {e}")

            except Exception as e:
                st.error(f"Classifier setup error: {e}")


# ══════════════════════════════════════════════════════════
#  PAGE: CHATBOT
# ══════════════════════════════════════════════════════════
elif page == "🤖 Chatbot":
    import random as _rnd

    st.markdown("""<div class="dm-pagehead"><div class="icon">🤖</div>
    <div><div class="title">Chatbot Studio</div>
    <div class="sub">Upload any CSV — map your columns — train and chat instantly</div></div></div>""",
    unsafe_allow_html=True)

    # ── Premium gate ──────────────────────────────────────
    if not _require_premium("Chatbot Training"):
        st.stop()

    st.markdown("""<style>
    .cb-wrap{display:flex;flex-direction:column;gap:0.65rem;padding:0.25rem 0 1rem;}
    .cb-row-u{display:flex;justify-content:flex-end;}
    .cb-row-b{display:flex;justify-content:flex-start;flex-direction:column;gap:2px;}
    .cb-bubble{padding:0.55rem 1rem;border-radius:14px;font-size:0.84rem;
      line-height:1.55;font-family:'IBM Plex Sans',sans-serif;
      max-width:70%;word-wrap:break-word;display:inline-block;}
    .cb-bu{background:#1a56db;color:#fff;border-bottom-right-radius:3px;}
    .cb-bb{background:#f1f5f9;color:#111827;border-bottom-left-radius:3px;}
    .cb-meta{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#94a3b8;padding-left:4px;}
    </style>""", unsafe_allow_html=True)

    cb_tab1, cb_tab2 = st.tabs(["🧠 Train", "💬 Chat"])

    # ══════════════════════════════════════════
    #  TRAIN TAB
    # ══════════════════════════════════════════
    with cb_tab1:
        st.markdown(
            '<div class="dm-card"><div class="dm-card-title">Upload Your CSV — Any Format Works</div>'
            '<p style="font-size:0.82rem;color:#374151;margin:0 0 0.5rem">'
            'Upload <strong>any CSV with at least 2 columns</strong>. '
            'Then select which column is the user input and which is the bot reply. '
            'An optional intent/category column improves accuracy.</p></div>',
            unsafe_allow_html=True)

        # Sample CSV download
        with st.expander("📥 Download sample CSVs"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                s1 = pd.DataFrame([
                    {"question": "Hello",                "answer": "Hi! How can I help you?"},
                    {"question": "Hi there",             "answer": "Hello! What can I do for you?"},
                    {"question": "What is your name?",   "answer": "I am DataMind Bot!"},
                    {"question": "How are you?",         "answer": "I am doing great, thanks for asking!"},
                    {"question": "What can you do?",     "answer": "I can answer questions based on my training."},
                    {"question": "Goodbye",              "answer": "Goodbye! Have a great day!"},
                    {"question": "Thank you",            "answer": "You are welcome!"},
                    {"question": "Help me",              "answer": "Sure! Tell me what you need."},
                    {"question": "Tell me a joke",       "answer": "Why did the ML model go to school? To improve its learning rate!"},
                    {"question": "What is DataMind?",    "answer": "DataMind AI is a production AutoML platform."},
                ])
                st.download_button("⬇ Q&A format (question/answer)",
                                   data=s1.to_csv(index=False).encode(),
                                   file_name="sample_qa.csv", mime="text/csv")
            with col_s2:
                s2 = pd.DataFrame([
                    {"intent": "greeting",  "utterance": "Hello",         "response": "Hi there! How can I help?"},
                    {"intent": "greeting",  "utterance": "Hi",            "response": "Hello! What do you need?"},
                    {"intent": "greeting",  "utterance": "Hey",           "response": "Hey! Nice to meet you."},
                    {"intent": "farewell",  "utterance": "Goodbye",       "response": "Bye! Have a great day!"},
                    {"intent": "farewell",  "utterance": "See you later",  "response": "Take care!"},
                    {"intent": "thanks",    "utterance": "Thank you",     "response": "You're welcome!"},
                    {"intent": "thanks",    "utterance": "Thanks a lot",  "response": "Happy to help!"},
                    {"intent": "help",      "utterance": "Help me",       "response": "Of course! What do you need?"},
                    {"intent": "help",      "utterance": "I need help",   "response": "I'm here. Tell me the issue."},
                    {"intent": "name",      "utterance": "What's your name?", "response": "I am DataMind Bot!"},
                ])
                st.download_button("⬇ Intent format (intent/utterance/response)",
                                   data=s2.to_csv(index=False).encode(),
                                   file_name="sample_intents.csv", mime="text/csv")

        cb_file = st.file_uploader("Upload CSV", type=["csv"], key="cb_up")

        if cb_file:
            try:
                cb_raw = pd.read_csv(io.BytesIO(cb_file.read()))
            except Exception as e:
                st.error(f"Could not read CSV: {e}"); st.stop()

            if len(cb_raw.columns) < 2:
                st.error("CSV must have at least 2 columns."); st.stop()

            all_cols_cb = list(cb_raw.columns)
            st.success(f"✅ Loaded {len(cb_raw)} rows · {len(all_cols_cb)} columns")
            st.dataframe(cb_raw.head(6), use_container_width=True)

            st.markdown("#### Map your columns")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                cb_in_col   = st.selectbox("📥 User input column",
                                            all_cols_cb, index=0, key="cb_in")
            with mc2:
                cb_out_col  = st.selectbox("📤 Bot response column",
                                            all_cols_cb,
                                            index=min(1, len(all_cols_cb)-1), key="cb_out")
            with mc3:
                cb_int_col  = st.selectbox("🏷 Intent column (optional — improves accuracy)",
                                            ["— none —"] + all_cols_cb, key="cb_int")

            mc4, mc5 = st.columns(2)
            with mc4: cb_mf = int(st.number_input("TF-IDF max features", value=3000, min_value=100, max_value=20000, step=100, key="cb_mf"))
            with mc5: cb_ng = st.selectbox("N-gram range", ["(1,1)", "(1,2)", "(1,3)"], index=1, key="cb_ng")
            cb_ng_p = tuple(int(x) for x in cb_ng.strip("()").split(","))

            # Show column mapping preview
            st.markdown(f"**Mapping preview** — input: `{cb_in_col}` → response: `{cb_out_col}`" +
                        (f" · intent: `{cb_int_col}`" if cb_int_col != "— none —" else ""))
            st.dataframe(cb_raw[[cb_in_col, cb_out_col] +
                                 ([cb_int_col] if cb_int_col != "— none —" else [])].dropna().head(6),
                         use_container_width=True)

            if st.button("🚀 Train Chatbot", key="cb_train_btn"):
                try:
                    with st.spinner("Building intent classifier…"):
                        # Build working dataframe
                        cb_work = cb_raw[[cb_in_col, cb_out_col]].copy()
                        cb_work.columns = ["_input", "_output"]

                        # Determine intent column
                        if cb_int_col != "— none —":
                            cb_work["_intent"] = cb_raw[cb_int_col].astype(str)
                        else:
                            # Use response as intent (each unique response = one intent)
                            cb_work["_intent"] = cb_work["_output"].astype(str)

                        cb_work = cb_work.dropna(subset=["_input", "_output"])
                        cb_work["_input"]  = cb_work["_input"].astype(str).str.strip()
                        cb_work["_output"] = cb_work["_output"].astype(str)
                        cb_work["_intent"] = cb_work["_intent"].astype(str)
                        cb_work = cb_work[cb_work["_input"].str.len() > 0]

                        if len(cb_work) < 2:
                            st.error("Not enough valid rows (need at least 2) after removing empty inputs.")
                            st.stop()

                        # Build response map: intent → list of responses
                        resp_map = (cb_work.groupby("_intent")["_output"]
                                    .apply(lambda x: list(set(x.tolist())))
                                    .to_dict())

                        le_cb  = LabelEncoder()
                        y_cb   = le_cb.fit_transform(cb_work["_intent"].tolist())

                        vec_cb = TfidfVectorizer(max_features=cb_mf, ngram_range=cb_ng_p,
                                                 sublinear_tf=True, strip_accents="unicode",
                                                 analyzer="word")
                        X_cb   = vec_cb.fit_transform(cb_work["_input"].tolist())

                        clf_cb = LogisticRegression(max_iter=500, C=1.5, solver="saga", n_jobs=N_JOBS)
                        clf_cb.fit(X_cb, y_cb)
                        tr_acc = float(accuracy_score(y_cb, clf_cb.predict(X_cb)))

                        st.session_state.chatbot_model      = clf_cb
                        st.session_state.chatbot_vectorizer = vec_cb
                        st.session_state.chatbot_responses  = resp_map
                        st.session_state.chatbot_classes    = le_cb.classes_.tolist()

                    n_intents = len(le_cb.classes_)
                    k1cb, k2cb, k3cb = st.columns(3)
                    k1cb.markdown(f'<div class="dm-kpi green"><div class="val">{tr_acc:.3f}</div><div class="lbl">Training Accuracy</div></div>', unsafe_allow_html=True)
                    k2cb.markdown(f'<div class="dm-kpi gold"><div class="val">{n_intents}</div><div class="lbl">Intents</div></div>', unsafe_allow_html=True)
                    k3cb.markdown(f'<div class="dm-kpi"><div class="val">{len(cb_work)}</div><div class="lbl">Training Examples</div></div>', unsafe_allow_html=True)
                    st.success(f"✅ Chatbot trained! Switch to the **Chat** tab to talk to it.")

                    # Intent distribution chart (top 20)
                    ic = cb_work["_intent"].value_counts().head(20)
                    if len(ic) > 1:
                        fig, ax = plt.subplots(figsize=(max(6, len(ic)*0.55), 3.5))
                        clrs = [C_BLUE if i%2==0 else C_GOLD for i in range(len(ic))]
                        bars = ax.bar(ic.index, ic.values, color=clrs, edgecolor="white")
                        ax.bar_label(bars, padding=3, fontsize=8)
                        ax.set_xlabel("Intent"); ax.set_ylabel("Examples")
                        ax.set_title("Training examples per intent")
                        plt.xticks(rotation=35, ha="right"); fig.tight_layout()
                        st.pyplot(fig); plt.close(fig)

                except Exception as e:
                    st.error(f"Training error: {e}")

        else:
            st.markdown('<div class="dm-upload"><div class="uicon">🤖</div>'
                        '<div class="hint">Upload any CSV — Q&A pairs, intents, support tickets, FAQs…</div></div>',
                        unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  CHAT TAB
    # ══════════════════════════════════════════
    with cb_tab2:
        if not st.session_state.chatbot_model:
            st.info("🤖 No chatbot trained yet. Go to the **Train** tab, upload a CSV, and train first.")
        else:
            n_cls_cb = len(st.session_state.chatbot_classes)
            st.markdown(f'<span class="dm-badge green">🟢 Chatbot ready · {n_cls_cb} intents</span>',
                        unsafe_allow_html=True)

            def _bot_reply(msg):
                """Return (response, intent, confidence)."""
                msg = msg.strip()
                if not msg:
                    return "Please type something!", "—", 0.0
                xv  = st.session_state.chatbot_vectorizer.transform([msg])
                try:
                    prbs   = st.session_state.chatbot_model.predict_proba(xv)[0]
                    best_i = int(np.argmax(prbs))
                    intent = st.session_state.chatbot_classes[best_i]
                    conf   = float(prbs[best_i])
                except AttributeError:
                    # LinearSVC fallback (no predict_proba)
                    pred   = int(st.session_state.chatbot_model.predict(xv)[0])
                    intent = st.session_state.chatbot_classes[pred]
                    conf   = 1.0

                if conf < 0.20:
                    return ("I'm not sure I understand that. Could you rephrase?",
                            "low_confidence", conf)

                resps = st.session_state.chatbot_responses.get(
                    intent, ["I don't have a response for that."])
                return _rnd.choice(resps), intent, conf

            # Render chat history
            if st.session_state.chat_history:
                html_chat = '<div class="cb-wrap">'
                for m in st.session_state.chat_history:
                    if m["role"] == "user":
                        html_chat += (f'<div class="cb-row-u">'
                                      f'<span class="cb-bubble cb-bu">{m["content"]}</span></div>')
                    else:
                        conf_s = f"{m.get('conf', 0):.0%}" if m.get("conf") is not None else ""
                        html_chat += (f'<div class="cb-row-b">'
                                      f'<span class="cb-bubble cb-bb">{m["content"]}</span>'
                                      f'<span class="cb-meta">intent: {m.get("intent","—")} · {conf_s}</span>'
                                      f'</div>')
                html_chat += '</div>'
                st.markdown(html_chat, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:center;color:#94a3b8;font-size:0.82rem;padding:2.5rem">'
                    'Send a message below to start chatting ↓</div>',
                    unsafe_allow_html=True)

            # Input row
            ci1, ci2, ci3 = st.columns([5, 1, 1])
            with ci1:
                user_msg = st.text_input("Your message", placeholder="Type here…",
                                          label_visibility="collapsed", key="cb_input_msg")
            with ci2:
                send_btn = st.button("Send", use_container_width=True, key="cb_send_btn")
            with ci3:
                if st.button("Clear", use_container_width=True, key="cb_clr_btn"):
                    st.session_state.chat_history = []
                    st.rerun()

            if send_btn and user_msg.strip():
                reply, det_intent, det_conf = _bot_reply(user_msg)
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                st.session_state.chat_history.append({
                    "role": "bot", "content": reply,
                    "intent": det_intent, "conf": det_conf
                })
                st.rerun()

            # Intent confidence expander for last message
            if len(st.session_state.chat_history) >= 2:
                last_u = next((m["content"] for m in reversed(st.session_state.chat_history)
                               if m["role"] == "user"), None)
                if last_u:
                    try:
                        with st.expander("🔍 Intent confidence breakdown"):
                            xv2    = st.session_state.chatbot_vectorizer.transform([last_u])
                            prbs2  = st.session_state.chatbot_model.predict_proba(xv2)[0]
                            clss2  = st.session_state.chatbot_classes
                            n_show = min(8, len(clss2))
                            tidx2  = np.argsort(prbs2)[-n_show:][::-1]
                            fig, ax = plt.subplots(figsize=(6, max(2.5, n_show*0.38)))
                            ax.barh([clss2[i] for i in tidx2],
                                    [float(prbs2[i]) for i in tidx2],
                                    color=[C_BLUE if k==0 else "#e5e7eb" for k in range(n_show)],
                                    edgecolor="white")
                            ax.set_xlim(0, 1)
                            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                            ax.set_title("Top intent probabilities")
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    except Exception:
                        pass   # LinearSVC doesn't support predict_proba — silently skip chart
elif page == "💳 Pricing":
    st.markdown("""
    <div class="dm-pagehead">
      <div class="icon">💳</div>
      <div><div class="title">Plans & Pricing</div>
      <div class="sub">Simple, transparent pricing for every user</div></div>
    </div>""", unsafe_allow_html=True)

    _is_premium  = st.session_state.auth_plan == "premium"
    _is_pending  = st.session_state.auth_plan == "pending_review"
    _used        = st.session_state.auth_proj_used
    _left        = max(0, PLAN_FREE_PROJ - _used)

    # ── Plan cards ──────────────────────────────────────────
    _pc1, _pc2 = st.columns(2)
    with _pc1:
        _border = "#1a56db" if (not _is_premium and not _is_pending) else "#e2e6ea"
        st.markdown(f"""
        <div class="dm-card" style="border-top:4px solid {_border};min-height:320px">
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.08em;color:#6b7280;margin-bottom:.5rem">Free Plan</div>
          <div style="font-size:2rem;font-weight:700;color:#111827">
            ₹0 <span style="font-size:0.9rem;font-weight:400;color:#6b7280">/ forever</span>
          </div>
          <hr style="border-color:#e2e6ea;margin:1rem 0">
          <ul style="font-size:0.85rem;line-height:2.2;color:#374151;padding-left:1.2rem">
            <li>✅ {PLAN_FREE_PROJ} projects total</li>
            <li>✅ Analysis, AutoML, Evaluation</li>
            <li>✅ NLP & Chatbot</li>
            <li>✅ Auto Labeling (TF-IDF)</li>
            <li>❌ Deep Learning / CNN / LSTM</li>
            <li>❌ Ollama hybrid labeling</li>
            <li>❌ Firebase persistence</li>
          </ul>
          {"<div style='margin-top:.5rem'><span style='background:#fef3c7;color:#b45309;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600'>Current Plan</span></div>" if (not _is_premium and not _is_pending) else ""}
        </div>""", unsafe_allow_html=True)

    with _pc2:
        st.markdown(f"""
        <div class="dm-card" style="border-top:4px solid #1a56db;min-height:320px;
          box-shadow:0 4px 24px rgba(26,86,219,.12)">
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.08em;color:#1a56db;margin-bottom:.5rem">
            Premium Plan &nbsp;⭐
          </div>
          <div style="font-size:2rem;font-weight:700;color:#111827">
            ₹{PLAN_PRICE} <span style="font-size:0.9rem;font-weight:400;color:#6b7280">/ month</span>
          </div>
          <hr style="border-color:#e2e6ea;margin:1rem 0">
          <ul style="font-size:0.85rem;line-height:2.2;color:#374151;padding-left:1.2rem">
            <li>✅ <strong>Unlimited</strong> projects</li>
            <li>✅ Everything in Free</li>
            <li>✅ Deep Learning · CNN · LSTM</li>
            <li>✅ Ollama hybrid auto-labeling</li>
            <li>✅ Firebase persistence</li>
            <li>✅ SHAP explainability</li>
            <li>✅ Priority support</li>
          </ul>
          {"<div style='margin-top:.5rem'><span style='background:#d1fae5;color:#047857;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600'>Current Plan — Active</span></div>" if _is_premium else ""}
          {"<div style='margin-top:.5rem'><span style='background:#fef3c7;color:#d97706;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600'>⏳ Pending Approval</span></div>" if _is_pending else ""}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

    # ── Usage meter ──────────────────────────────────────────
    st.markdown('<div class="dm-card-title">Your Usage</div>', unsafe_allow_html=True)
    _um1, _um2, _um3 = st.columns(3)
    _um1.markdown(
        f'<div class="dm-kpi"><div class="val">{_used}</div>'
        f'<div class="lbl">Projects Used</div></div>', unsafe_allow_html=True)
    _um2.markdown(
        f'<div class="dm-kpi {"gold" if _left == 0 and not _is_premium else "green"}">'
        f'<div class="val">{"∞" if _is_premium else ("—" if _is_pending else _left)}</div>'
        f'<div class="lbl">Projects Left</div></div>', unsafe_allow_html=True)
    _plan_disp  = "Premium" if _is_premium else ("⏳ Pending" if _is_pending else "Free")
    _plan_clr   = "#047857" if _is_premium else ("#d97706" if _is_pending else "#b45309")
    _um3.markdown(
        f'<div class="dm-kpi"><div class="val" style="color:{_plan_clr}">'
        f'{_plan_disp}</div>'
        f'<div class="lbl">Current Plan</div></div>', unsafe_allow_html=True)

    if _is_premium and st.session_state.auth_paid_until:
        _days_rem = (st.session_state.auth_paid_until - datetime.datetime.utcnow()).days
        st.info(f"Your Premium plan expires in **{max(0, _days_rem)} days** "
                f"({st.session_state.auth_paid_until.strftime('%d %b %Y')}). "
                f"Pay again before expiry to continue uninterrupted.")

    if _is_pending:
        st.warning(
            "⏳ **Payment under review.** Your UTR has been received and is being verified "
            "by our team. Premium access will be granted within 24 hours. "
            "Contact support if it takes longer.")

    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

    # ── Payment section — only show to free users ────────────────────────
    if not _is_premium and not _is_pending:
        st.markdown("### Upgrade Now — ₹300/month")

        # Check for Razorpay callback on this page too
        if _check_razorpay_callback():
            st.success("Payment verified! Premium activated.")
            st.balloons()
            st.rerun()

        _pay1, _pay2 = st.columns(2)

        with _pay1:
            st.markdown("""
            <div class="dm-card">
              <div class="dm-card-title">How to Pay</div>
              <ol style="font-size:0.85rem;line-height:2.4;color:#374151;padding-left:1.2rem">
                <li><strong>Razorpay (Recommended):</strong> Click Pay button below — instant activation</li>
                <li><strong>UPI Manual:</strong> Scan QR or use UPI ID, then enter 12-digit UTR</li>
                <li>Activation is instant for Razorpay; within 24 h for manual UPI</li>
              </ol>
            </div>""", unsafe_allow_html=True)

        with _pay2:
            st.markdown('<div class="dm-card"><div class="dm-card-title">UPI QR Code</div>',
                        unsafe_allow_html=True)
            _qr2 = _generate_upi_qr()
            if _qr2:
                import io as _bio2
                _buf2 = _bio2.BytesIO()
                _qr2.save(_buf2, format="PNG")
                st.image(_buf2.getvalue(), width=180, caption="Pay ₹" + str(PLAN_PRICE) + " via UPI")
            else:
                st.code("UPI ID: " + UPI_ID, language=None)
            st.markdown(
                '<div style="background:#f0f7ff;border-radius:8px;padding:.6rem 1rem;'
                'font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#1a56db">'
                + UPI_ID + '<br>&#8377;' + str(PLAN_PRICE) + ' &middot; ' + UPI_NAME +
                '</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

        _ptab2_rzp, _ptab2_upi = st.tabs(["💳 Razorpay — Instant Activation", "🏦 UPI — Manual Verification"])

        with _ptab2_rzp:
            st.markdown(
                '<div style="background:#f0f7ff;border:1px solid #bdd3fc;border-radius:10px;'
                'padding:.7rem 1rem;font-size:.82rem;color:#1e40af;margin-bottom:.75rem">'
                '&#9889; Razorpay handles cards, UPI, netbanking, wallets. '
                'Premium activates <strong>immediately</strong> after payment.</div>',
                unsafe_allow_html=True)
            if "rzp_order_pricing" not in st.session_state:
                try:
                    st.session_state.rzp_order_pricing = _razorpay_create_order(PLAN_PRICE)
                except Exception as _ep:
                    st.session_state.rzp_order_pricing = None
                    st.warning("Razorpay unavailable: " + str(_ep))
            _rzp_o2 = st.session_state.get("rzp_order_pricing")
            if _rzp_o2 and _rzp_o2.get("id"):
                _render_razorpay_button(
                    order_id=_rzp_o2["id"],
                    amount_inr=PLAN_PRICE,
                    user_email=st.session_state.auth_email,
                )
            else:
                st.info("Add RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET in main.py to enable this.")

        with _ptab2_upi:
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;'
                'padding:.55rem .85rem;font-size:.78rem;color:#92400e;margin-bottom:.65rem">'
                '&#9888; Paste your 12-digit UTR from your UPI app after paying. '
                'Admin verifies and activates within 24 hours.</div>',
                unsafe_allow_html=True)
            _utr2 = st.text_input("UTR / Transaction ID",
                                   placeholder="12-digit UTR number", key="pricing_utr")
            if st.button("Submit for Verification", key="pricing_activate"):
                _ok2, _msg2 = _submit_payment(st.session_state.auth_uid, _utr2)
                if _ok2:
                    st.success(_msg2)
                    st.session_state.pop("rzp_order_pricing", None)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(_msg2)

    elif _is_premium:
        st.success("You are on the Premium plan. Enjoy unlimited access!")
        if st.button("Renew Premium (₹300)", key="pricing_renew"):
            st.session_state.pop("rzp_order", None)
            st.session_state.pop("rzp_order_pricing", None)
            st.session_state.auth_page = "payment"
            st.rerun()
elif page == "🤖 Auto Labeling":
    import requests as _rq

    # ════════════════════════════════════════════════════════════
    #  SHARED STYLES — Auto Labeling Studio
    # ════════════════════════════════════════════════════════════
    st.markdown("""
    <style>
    .als-hero{background:linear-gradient(135deg,#0f1b2d 0%,#1a3a6b 60%,#1a56db 100%);
      border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;
      position:relative;overflow:hidden;}
    .als-hero::after{content:'';position:absolute;width:300px;height:300px;
      border-radius:50%;background:radial-gradient(circle,rgba(96,165,250,.2),transparent 70%);
      top:-80px;right:-60px;pointer-events:none;}
    .als-hero-title{font-size:1.45rem;font-weight:800;color:#fff;
      letter-spacing:-.02em;margin-bottom:.3rem;}
    .als-hero-sub{font-size:.8rem;color:#93c5fd;margin-bottom:1.25rem;}
    .als-hero-chips{display:flex;gap:.5rem;flex-wrap:wrap;}
    .als-chip{padding:.22rem .8rem;border-radius:20px;font-size:.68rem;font-weight:600;
      background:rgba(255,255,255,.12);color:#bfdbfe;
      border:1px solid rgba(255,255,255,.18);}
    .als-chip.active{background:rgba(255,255,255,.22);color:#fff;}
    .als-mode-card{background:#fff;border:2px solid #e2e6ea;border-radius:14px;
      padding:1.5rem;cursor:pointer;transition:all .2s;text-align:center;}
    .als-mode-card:hover{border-color:#1a56db;
      box-shadow:0 4px 20px rgba(26,86,219,.12);}
    .als-mode-card.selected{border-color:#1a56db;
      background:linear-gradient(135deg,#eef4ff,#f0f7ff);}
    .als-mode-icon{font-size:2rem;margin-bottom:.5rem;}
    .als-mode-title{font-size:.95rem;font-weight:700;color:#0f1b2d;margin-bottom:.25rem;}
    .als-mode-sub{font-size:.75rem;color:#64748b;}
    .als-stat{background:#fff;border:1px solid #e2e6ea;border-radius:12px;
      padding:.85rem 1rem;text-align:center;}
    .als-stat-val{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;
      font-weight:700;color:#1a56db;}
    .als-stat-val.green{color:#047857;}
    .als-stat-val.gold{color:#b45309;}
    .als-stat-val.purple{color:#6d28d9;}
    .als-stat-lbl{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;
      color:#6b7280;margin-top:2px;}
    .als-pipeline-bar{display:flex;align-items:center;gap:0;
      background:#f8fafc;border:1px solid #e2e6ea;border-radius:10px;
      overflow:hidden;margin:1rem 0;}
    .als-pipe-step{flex:1;padding:.55rem .5rem;text-align:center;
      font-size:.7rem;font-weight:600;color:#64748b;position:relative;}
    .als-pipe-step.done{background:#d1fae5;color:#047857;}
    .als-pipe-step.active{background:#1a56db;color:#fff;}
    .als-pipe-step.pending{background:#f8fafc;color:#94a3b8;}
    .als-img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
      gap:12px;margin-top:1rem;}
    .als-img-card{background:#fff;border:1.5px solid #e2e6ea;border-radius:12px;
      overflow:hidden;transition:all .2s;}
    .als-img-card:hover{border-color:#1a56db;
      box-shadow:0 4px 16px rgba(26,86,219,.12);}
    .als-img-label{padding:.5rem .6rem;font-size:.72rem;font-weight:600;
      color:#0f1b2d;border-top:1px solid #f1f5f9;}
    .als-img-conf{font-size:.65rem;color:#6b7280;}
    .als-section{background:#fff;border:1px solid #e2e6ea;border-radius:14px;
      padding:1.25rem 1.5rem;margin-bottom:1rem;}
    .als-section-title{font-size:.72rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.08em;color:#94a3b8;margin-bottom:1rem;
      padding-bottom:.6rem;border-bottom:1px solid #f1f5f9;}
    </style>
    """, unsafe_allow_html=True)

    # ── Session state ───────────────────────────────────────────
    for _k,_v in [
        ("al_mode_type","text"),          # "text" | "image"
        ("al_df",None),("al_results",None),("al_labels",[]),
        ("al_text_col",None),("al_mode","supervised"),
        ("al_review_df",None),("al_trained_model",None),
        ("al_trained_vec",None),("al_seed_data",{}),("al_n_clusters",5),
        ("al_ollama_model","phi3"),("al_use_ollama",True),
        ("al_ollama_thresh",0.72),
        ("al_use_gemini",False),("al_gemini_model","gemini-1.5-flash"),
        # Image labeling
        ("al_img_files",[]),("al_img_results",[]),
        ("al_yolo_model","yolov8n.pt"),("al_img_conf",0.4),
        ("al_img_labels",[]),
    ]:
        if _k not in st.session_state: st.session_state[_k] = _v

    # ── Ollama helpers ──────────────────────────────────────────
    _OB = "http://localhost:11434"
    _OM = ["phi3","llama3"]

    def _opng():
        try:
            return _rq.get(f"{_OB}/api/tags",timeout=3).status_code==200
        except: return False

    def _omodels():
        try:
            return [m["name"].split(":")[0]
                    for m in _rq.get(f"{_OB}/api/tags",timeout=3).json().get("models",[])]
        except: return []

    def _olabel(text, labels, model="phi3", timeout=30):
        _ls = ", ".join(f'"{l}"' for l in labels)
        _pr = (f"Classify the text into EXACTLY ONE label: {_ls}.\n"
               f"Reply with ONLY the label name.\nText: {text[:400]}\nLabel:")
        try:
            _r = _rq.post(f"{_OB}/api/generate",
                json={"model":model,"prompt":_pr,"stream":False,
                      "options":{"temperature":0.0,"num_predict":12}},timeout=timeout)
            raw = _r.json().get("response","").strip().lower().strip('"\'')
            for _l in labels:
                if raw == _l.lower(): return _l, 0.95
            for _l in labels:
                if _l.lower() in raw: return _l, 0.80
            return labels[0], 0.55
        except: return labels[0], 0.40

    # ── Gemini helpers ─────────────────────────────────────────────
    _GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
    _GEMINI_URL    = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _gemini_label(text, labels, model="gemini-1.5-flash", api_key="", timeout=20):
        """Call Gemini REST API to classify text. Returns (label, confidence)."""
        if not api_key:
            return labels[0], 0.40
        _ls  = ", ".join(f'"{l}"' for l in labels)
        _pr  = (f"Classify the text into EXACTLY ONE label: {_ls}.\n"
                f"Reply with ONLY the label name, nothing else.\n"
                f"Text: {text[:500]}\nLabel:")
        _url = _GEMINI_URL.format(model=model) + "?key=" + api_key
        try:
            _r = _rq.post(
                _url,
                json={"contents": [{"parts": [{"text": _pr}]}],
                      "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16}},
                timeout=timeout,
            )
            _r.raise_for_status()
            raw = (_r.json()
                     .get("candidates", [{}])[0]
                     .get("content", {})
                     .get("parts", [{}])[0]
                     .get("text", "")
                     .strip().lower().strip('"\' '))
            for _l in labels:
                if raw == _l.lower(): return _l, 0.96
            for _l in labels:
                if _l.lower() in raw: return _l, 0.82
            return labels[0], 0.55
        except Exception:
            return labels[0], 0.40

    def _test_gemini(api_key, model):
        """Quick connectivity test. Returns (ok, message)."""
        if not api_key:
            return False, "No API key set. Get one at aistudio.google.com/app/apikey"
        try:
            _url = _GEMINI_URL.format(model=model) + "?key=" + api_key
            _r = _rq.post(
                _url,
                json={"contents": [{"parts": [{"text": "Reply OK only."}]}],
                      "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4}},
                timeout=10,
            )
            if _r.status_code == 200: return True, "Gemini connected!"
            elif _r.status_code == 400: return False, "Invalid API key or model."
            elif _r.status_code == 429: return False, "Rate limit — wait a moment."
            else: return False, f"HTTP {_r.status_code}: {_r.text[:100]}"
        except Exception as _te:
            return False, str(_te)[:100]

    # ════════════════════════════════════════════════════════════
    #  HERO BANNER
    # ════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="als-hero">
      <div class="als-hero-title">&#127991; Auto Labeling Studio</div>
      <div class="als-hero-sub">
        Scale AI-style intelligent labeling &middot;
        TF-IDF fast pass &rarr; Gemini / Ollama LLM refinement &middot;
        Offline YOLO detection &middot; Human-in-the-loop review
      </div>
      <div class="als-hero-chips">
        <span class="als-chip active">Text Labeling</span>
        <span class="als-chip active">Image Labeling</span>
        <span class="als-chip active">Gemini AI</span>
        <span class="als-chip active">Ollama LLM</span>
        <span class="als-chip active">Offline YOLO</span>
        <span class="als-chip active">Human Review</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  MODE SELECTOR
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="als-section-title" style="font-size:.8rem;font-weight:700;'
                'color:#374151;text-transform:none;letter-spacing:0">Select Labeling Mode</div>',
                unsafe_allow_html=True)
    _mc1, _mc2 = st.columns(2)
    with _mc1:
        _txt_sel = st.session_state.al_mode_type == "text"
        st.markdown(
            '<div class="als-mode-card' + (' selected' if _txt_sel else '') + '">'
            '<div class="als-mode-icon">&#128203;</div>'
            '<div class="als-mode-title">Text Labeling</div>'
            '<div class="als-mode-sub">CSV / tabular data &middot; TF-IDF + Ollama hybrid'
            ' &middot; Supervised &amp; unsupervised</div>'
            '</div>', unsafe_allow_html=True)
        if st.button("Select Text Mode", key="al_mode_txt",
                     use_container_width=True):
            st.session_state.al_mode_type = "text"; st.rerun()
    with _mc2:
        _img_sel = st.session_state.al_mode_type == "image"
        st.markdown(
            '<div class="als-mode-card' + (' selected' if _img_sel else '') + '">'
            '<div class="als-mode-icon">&#128247;</div>'
            '<div class="als-mode-title">Image Labeling</div>'
            '<div class="als-mode-sub">Upload images &middot; YOLO v8 object detection'
            ' &middot; Bounding boxes &amp; class labels</div>'
            '</div>', unsafe_allow_html=True)
        if st.button("Select Image Mode", key="al_mode_img",
                     use_container_width=True):
            st.session_state.al_mode_type = "image"; st.rerun()

    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  TEXT LABELING
    # ════════════════════════════════════════════════════════════
    if st.session_state.al_mode_type == "text":
        al_tab1, al_tab2, al_tab3, al_tab4 = st.tabs(
            ["⚙️  Setup", "⚡  Auto-Label", "👁️  Review Queue", "📥  Export"])

        # ── TAB 1 SETUP ──────────────────────────────────────────
        with al_tab1:
            _s1 = st.container()
            with _s1:
                st.markdown('<div class="als-section">'
                            '<div class="als-section-title">LLM Configuration — Gemini or Ollama</div>',
                            unsafe_allow_html=True)
                _is_prem = st.session_state.auth_plan == "premium"

                # ── LLM selector ───────────────────────────────────────
                _llm_choice = st.radio(
                    "LLM backend for Phase 2 refinement",
                    ["None (TF-IDF only)", "Gemini (cloud, free tier)", "Ollama (local)"],
                    horizontal=True, key="al_llm_choice",
                    help="Gemini works on Streamlit Cloud. Ollama needs a local server."
                )
                _oon  = False
                _ugem = False

                # ── Gemini config ───────────────────────────────────────
                if "Gemini" in _llm_choice:
                    _ugem = True
                    st.session_state.al_use_gemini = True
                    st.session_state.al_use_ollama = False
                    _gc1, _gc2, _gc3 = st.columns([2, 2, 1])
                    with _gc1:
                        _gkey = st.text_input(
                            "Gemini API Key",
                            value=GEMINI_API_KEY or st.session_state.get("al_gemini_key", ""),
                            type="password",
                            placeholder="AIza...",
                            key="al_gemini_key_inp",
                            help="Get free key at aistudio.google.com/app/apikey"
                        )
                        st.session_state["al_gemini_key"] = _gkey
                    with _gc2:
                        _gmodel = st.selectbox("Model", _GEMINI_MODELS,
                                               key="al_gm_sel",
                                               help="flash=fast+free, pro=smarter")
                        st.session_state.al_gemini_model = _gmodel
                    with _gc3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Test", key="al_gtest"):
                            _gok, _gmsg = _test_gemini(_gkey, _gmodel)
                            if _gok: st.success(_gmsg)
                            else:    st.error(_gmsg)
                    st.markdown(
                        '<div style="background:#f0fdf4;border:1px solid #86efac;'
                        'border-radius:8px;padding:.5rem .85rem;font-size:.78rem;'
                        'color:#166534;margin-top:.4rem">'
                        '&#9989; Gemini 1.5 Flash: 15 RPM free &middot; '
                        '1 million tokens/day &middot; Works on Streamlit Cloud</div>',
                        unsafe_allow_html=True)

                # ── Ollama config ───────────────────────────────────────
                elif "Ollama" in _llm_choice:
                    if not _is_prem:
                        st.markdown('<span class="dm-badge gold">Ollama — Premium only</span>',
                                    unsafe_allow_html=True)
                    else:
                        _oon = True
                        st.session_state.al_use_ollama = True
                        st.session_state.al_use_gemini = False
                        _oc2, _oc3 = st.columns([2, 1])
                        with _oc2:
                            _om = st.selectbox("Model", _OM, key="al_om_sel")
                            st.session_state.al_ollama_model = _om
                        with _oc3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Test", key="al_ping"):
                                if _opng():
                                    _av = _omodels()
                                    if st.session_state.al_ollama_model in _av:
                                        st.success("Ollama ready")
                                    else:
                                        st.warning(f"Run: ollama pull {st.session_state.al_ollama_model}")
                                else:
                                    st.error("Ollama offline — run: ollama serve")
                else:
                    st.session_state.al_use_ollama = False
                    st.session_state.al_use_gemini = False

                # ── Shared threshold ────────────────────────────────────
                if _oon or _ugem:
                    _oth = st.slider(
                        "Handoff threshold — rows below this confidence go to LLM",
                        0.40, 0.99, st.session_state.al_ollama_thresh, 0.01,
                        key="al_oth_sl")
                    st.session_state.al_ollama_thresh = _oth
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="als-section">'
                        '<div class="als-section-title">Upload Data</div>',
                        unsafe_allow_html=True)
            _af = st.file_uploader("Upload CSV", type="csv", key="al_up")
            if _af:
                _raw = load_csv(_af.read())
                st.session_state.al_df = _raw
                st.markdown(
                    f'<span class="dm-badge green">{len(_raw):,} rows &middot;'
                    f' {len(_raw.columns)} cols</span>', unsafe_allow_html=True)
                st.dataframe(_raw.head(4), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.al_df is not None:
                _df = st.session_state.al_df
                st.markdown('<div class="als-section">'
                            '<div class="als-section-title">Configure Task</div>',
                            unsafe_allow_html=True)
                _ca,_cb = st.columns(2)
                with _ca:
                    _tc = st.selectbox("Text column", _df.columns.tolist(), key="al_tc")
                    st.session_state.al_text_col = _tc
                with _cb:
                    _ms = st.radio("Mode",
                        ["Supervised (seed labels)","Unsupervised (auto-cluster)"],
                        key="al_ms")
                    st.session_state.al_mode = "supervised" if "Supervised" in _ms else "unsupervised"
                st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.al_mode == "supervised":
                    st.markdown('<div class="als-section">'
                                '<div class="als-section-title">Label Taxonomy</div>',
                                unsafe_allow_html=True)
                    _lr = st.text_area("Labels (one per line)",
                        value="positive\nnegative\nneutral", height=110, key="al_lta")
                    _labels = [l.strip() for l in _lr.splitlines() if l.strip()]
                    st.session_state.al_labels = _labels
                    if _labels:
                        st.markdown(" ".join(
                            f'<span class="dm-badge blue">{l}</span>' for l in _labels),
                            unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="als-section">'
                                '<div class="als-section-title">Seed Examples (optional)</div>',
                                unsafe_allow_html=True)
                    _seed = {}
                    for _lbl in _labels:
                        _seed[_lbl] = st.text_area(
                            f"Examples for {_lbl}",
                            placeholder=f"One per line for '{_lbl}'",
                            height=70, key=f"al_seed_{_lbl}")
                    st.session_state.al_seed_data = _seed
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="als-section">'
                                '<div class="als-section-title">Cluster Settings</div>',
                                unsafe_allow_html=True)
                    _nc = st.slider("Number of clusters",2,20,5,key="al_nc")
                    st.session_state.al_n_clusters = _nc
                    st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 2 AUTO-LABEL ─────────────────────────────────────
        with al_tab2:
            if st.session_state.al_df is None:
                st.markdown(
                    '<div style="text-align:center;padding:3rem;color:#94a3b8;">'
                    '<div style="font-size:2.5rem;margin-bottom:.5rem">&#128203;</div>'
                    '<div style="font-size:.9rem">Upload a CSV in the Setup tab first</div>'
                    '</div>', unsafe_allow_html=True)
            elif not st.session_state.al_text_col:
                st.warning("Select a text column in the Setup tab first.")
            elif st.session_state.al_mode=="supervised" and not st.session_state.al_labels:
                st.warning("Define at least one label in the Setup tab.")
            else:
                _df2 = st.session_state.al_df.copy()
                _tc2 = st.session_state.al_text_col
                _uol  = st.session_state.al_use_ollama
                _ugem = st.session_state.get("al_use_gemini", False)
                _gkey = st.session_state.get("al_gemini_key", "") or GEMINI_API_KEY

                if _ugem:
                    _gmod = st.session_state.get("al_gemini_model", "gemini-1.5-flash")
                    st.markdown(
                        '<div style="background:#f0fdf4;border:1px solid #86efac;'
                        'border-radius:10px;padding:.7rem 1rem;font-size:.8rem;'
                        'color:#166534;margin-bottom:1rem;">'
                        '<strong>Gemini hybrid pipeline active</strong> &mdash; '
                        'TF-IDF labels all rows &rarr; low-confidence rows sent to '
                        f'<strong>{_gmod}</strong></div>', unsafe_allow_html=True)
                elif _uol:
                    st.markdown(
                        '<div style="background:#eef4ff;border:1px solid #bdd3fc;'
                        'border-radius:10px;padding:.7rem 1rem;font-size:.8rem;'
                        'color:#1a56db;margin-bottom:1rem;">'
                        '<strong>Ollama hybrid pipeline active</strong> &mdash; '
                        'TF-IDF labels all rows &rarr; low-confidence rows sent to '
                        f'<strong>{st.session_state.al_ollama_model}</strong>'
                        '</div>', unsafe_allow_html=True)

                _ct = st.slider("Auto-accept threshold",0.5,1.0,0.80,0.01,key="al_ct")

                if st.button("Run Auto-Labeling", key="al_run",
                             use_container_width=True):
                    _texts = _df2[_tc2].fillna("").astype(str).tolist()
                    _mode2 = st.session_state.al_mode
                    _preds = [""]*len(_texts)
                    _confs = [0.0]*len(_texts)
                    _srcs  = ["tfidf"]*len(_texts)
                    try:
                        _ph1 = st.status("Phase 1 — TF-IDF fast pass", expanded=True)
                        with _ph1:
                            if _mode2 == "supervised":
                                _lbls = st.session_state.al_labels
                                _seeds = st.session_state.get("al_seed_data",{})
                                _tx,_ty = [],[]
                                for _l in _lbls:
                                    for _e in [s.strip() for s in _seeds.get(_l,"").splitlines() if s.strip()]:
                                        _tx.append(_e); _ty.append(_l)
                                for _l in _lbls: _tx.append(_l); _ty.append(_l)
                                _v2 = TfidfVectorizer(max_features=10000,ngram_range=(1,2),sublinear_tf=True)
                                _v2.fit(_texts+_tx)
                                _cl = LogisticRegression(max_iter=1000,C=2.0,solver="saga",n_jobs=N_JOBS)
                                _cl.fit(_v2.transform(_tx),_ty)
                                _pr = _cl.predict_proba(_v2.transform(_texts))
                                _preds = list(_cl.classes_[_pr.argmax(axis=1)])
                                _confs = list(_pr.max(axis=1).astype(float))
                                st.session_state.al_trained_model = _cl
                                st.session_state.al_trained_vec   = _v2
                            else:
                                from sklearn.cluster import KMeans
                                from sklearn.preprocessing import normalize as _nm
                                _nc2 = st.session_state.get("al_n_clusters",5)
                                _v2 = TfidfVectorizer(max_features=6000,ngram_range=(1,2),sublinear_tf=True)
                                _Xa = _v2.fit_transform(_texts)
                                _km = KMeans(n_clusters=_nc2,random_state=42,n_init=12)
                                _km.fit(_Xa)
                                _preds = [f"cluster_{c}" for c in _km.labels_]
                                _lbls  = [f"cluster_{i}" for i in range(_nc2)]
                                st.session_state.al_labels = _lbls
                                _Xd = _Xa.toarray()
                                _Xn = _nm(_Xd); _Cn = _nm(_km.cluster_centers_)
                                _confs = list((_Xn@_Cn.T).max(axis=1).astype(float))
                            _nlo = sum(1 for c in _confs if c < st.session_state.al_ollama_thresh)
                            st.write(f"TF-IDF done — {len(_texts)-_nlo:,} high-conf, {_nlo:,} to Ollama")
                        _ph1.update(label="Phase 1 complete",state="complete")

                        _run_phase2 = (_ugem or _uol) and _mode2 == "supervised" and _nlo > 0
                        if _run_phase2:
                            _ot2  = st.session_state.al_ollama_thresh
                            _low  = [i for i, c in enumerate(_confs) if c < _ot2]

                            if _ugem:
                                # ── Phase 2: Gemini ──────────────────────────
                                if not _gkey:
                                    st.warning("No Gemini API key — skipping Phase 2. "
                                               "Add it in the Setup tab.")
                                else:
                                    _gmod = st.session_state.get("al_gemini_model", "gemini-1.5-flash")
                                    _ph2 = st.status(
                                        f"Phase 2 — Gemini ({_gmod}) re-labeling {len(_low):,} rows",
                                        expanded=True)
                                    with _ph2:
                                        _pg = st.progress(0, "Starting Gemini...")
                                        _gem_errors = 0
                                        for _ii, _idx in enumerate(_low):
                                            _nl, _nc = _gemini_label(
                                                _texts[_idx],
                                                st.session_state.al_labels,
                                                model=_gmod,
                                                api_key=_gkey,
                                            )
                                            _preds[_idx] = _nl
                                            _confs[_idx] = _nc
                                            _srcs[_idx]  = f"gemini:{_gmod}"
                                            _pg.progress(
                                                (_ii + 1) / len(_low),
                                                text=f"Row {_ii+1}/{len(_low)} → {_nl} ({_nc:.0%})")
                                        st.write(f"Gemini labeled {len(_low):,} rows")
                                    _ph2.update(label="Phase 2 (Gemini) complete", state="complete")

                            elif _uol:
                                # ── Phase 2: Ollama ──────────────────────────
                                _om2 = st.session_state.al_ollama_model
                                _ph2 = st.status(
                                    f"Phase 2 — Ollama ({_om2}) re-labeling {len(_low):,} rows",
                                    expanded=True)
                                with _ph2:
                                    _pg = st.progress(0, "Starting Ollama...")
                                    if not _opng():
                                        st.warning("Ollama offline — skipping Phase 2")
                                    elif _om2 not in _omodels():
                                        st.warning(f"Run: ollama pull {_om2}")
                                    else:
                                        for _ii, _idx in enumerate(_low):
                                            _nl, _nc = _olabel(
                                                _texts[_idx],
                                                st.session_state.al_labels,
                                                model=_om2)
                                            _preds[_idx] = _nl
                                            _confs[_idx] = _nc
                                            _srcs[_idx]  = f"ollama:{_om2}"
                                            _pg.progress(
                                                (_ii + 1) / len(_low),
                                                text=f"Row {_ii+1}/{len(_low)} → {_nl}")
                                        st.write(f"Ollama labeled {len(_low):,} rows")
                                _ph2.update(label="Phase 2 (Ollama) complete", state="complete")

                        _res = _df2.copy()
                        _res["__predicted_label"] = _preds
                        _res["__confidence"]      = np.round(np.array(_confs,dtype=float),4)
                        _res["__source"]          = _srcs
                        _res["__status"]          = [
                            "auto-accepted" if c>=_ct else "needs review"
                            for c in _confs]
                        st.session_state.al_results   = _res
                        st.session_state.al_review_df = _res.copy()
                        st.success("Auto-labeling complete! Check the Review Queue tab.")
                    except Exception as _err:
                        st.error(f"Error: {_err}")

                if st.session_state.al_results is not None:
                    _r = st.session_state.al_results
                    _s1c,_s2c,_s3c,_s4c,_s5c = st.columns(5)
                    _nacc = (_r["__status"]=="auto-accepted").sum()
                    _nrev = (_r["__status"]=="needs review").sum()
                    _noll = int((_r["__source"].str.startswith("ollama") |
                                 _r["__source"].str.startswith("gemini")).sum()) if "__source" in _r.columns else 0
                    for _col,_val,_lbl,_cls in [
                        (_s1c,len(_r),"Total",""),
                        (_s2c,_nacc,"Accepted","green"),
                        (_s3c,_nrev,"Review","gold"),
                        (_s4c,f'{float(_r["__confidence"].mean()):.0%}',"Avg Conf",""),
                        (_s5c,_noll,"LLM Labeled","purple"),
                    ]:
                        _col.markdown(
                            f'<div class="als-stat"><div class="als-stat-val {_cls}">{_val}</div>'
                            f'<div class="als-stat-lbl">{_lbl}</div></div>',
                            unsafe_allow_html=True)
                    _ch1,_ch2,_ch3 = st.columns(3)
                    with _ch1:
                        _vc = _r["__predicted_label"].value_counts()
                        _fg,_ax = plt.subplots(figsize=(4,max(2.5,len(_vc)*.4)))
                        _ax.barh(_vc.index,_vc.values,color=C_BLUE,edgecolor="white")
                        _ax.set_title("Label Distribution"); _fg.tight_layout()
                        st.pyplot(_fg); plt.close(_fg)
                    with _ch2:
                        _fg2,_ax2 = plt.subplots(figsize=(4,3))
                        _ax2.hist(_r["__confidence"],bins=20,color=C_BLUE,alpha=.8,edgecolor="white")
                        _ax2.axvline(_ct,color=C_RED,lw=1.5,linestyle="--",label=f"Threshold")
                        if _uol: _ax2.axvline(st.session_state.al_ollama_thresh,color=C_PURPLE,lw=1.5,linestyle=":")
                        _ax2.legend(fontsize=8); _ax2.set_title("Confidence Dist"); _fg2.tight_layout()
                        st.pyplot(_fg2); plt.close(_fg2)
                    with _ch3:
                        if "__source" in _r.columns:
                            _sc = _r["__source"].value_counts()
                            _fg3,_ax3 = plt.subplots(figsize=(4,3))
                            _ax3.pie(_sc.values,labels=[s.replace("tfidf","TF-IDF").replace("ollama:","LLM:") for s in _sc.index],
                                     colors=[C_BLUE,C_PURPLE,C_GOLD,C_GREEN][:len(_sc)],
                                     autopct="%1.0f%%",startangle=90)
                            _ax3.set_title("Source Split"); _fg3.tight_layout()
                            st.pyplot(_fg3); plt.close(_fg3)

        # ── TAB 3 REVIEW QUEUE ───────────────────────────────────
        with al_tab3:
            if st.session_state.al_review_df is None:
                st.info("Run Auto-Labeling first (Tab 2) to populate the review queue.")
            else:
                _rdf = st.session_state.al_review_df
                _tc3 = st.session_state.al_text_col
                _lb3 = st.session_state.al_labels
                _rf1,_rf2,_rf3 = st.columns(3)
                with _rf1:
                    _sh = st.selectbox("Show",["All","Needs Review","Auto-Accepted"],key="al_shf")
                with _rf2:
                    _so = st.selectbox("Sort by",["Confidence (low first)","Confidence (high first)","Row order"],key="al_sof")
                with _rf3:
                    _lf = st.selectbox("Filter label",["All"]+list(_lb3),key="al_lff")
                _vw = _rdf.copy()
                if _sh == "Needs Review": _vw = _vw[_vw["__status"]=="needs review"]
                elif _sh == "Auto-Accepted": _vw = _vw[_vw["__status"]=="auto-accepted"]
                if _lf != "All": _vw = _vw[_vw["__predicted_label"]==_lf]
                if "low" in _so: _vw = _vw.sort_values("__confidence",ascending=True)
                elif "high" in _so: _vw = _vw.sort_values("__confidence",ascending=False)
                st.caption(f"Showing **{len(_vw):,}** of **{len(_rdf):,}** rows")
                _ecols = [c for c in [_tc3,"__predicted_label","__confidence","__source","__status"] if c in _rdf.columns]
                _edf = _vw[_ecols].reset_index(drop=False).rename(columns={"index":"_row_id"})
                _ed = st.data_editor(_edf, column_config={
                    "_row_id": st.column_config.NumberColumn("Row",disabled=True,width="small"),
                    _tc3: st.column_config.TextColumn("Text",disabled=True,width="large"),
                    "__predicted_label": st.column_config.SelectboxColumn("Label",options=_lb3,width="medium"),
                    "__confidence": st.column_config.NumberColumn("Conf",format="%.3f",disabled=True,width="small"),
                    "__source": st.column_config.TextColumn("Source",disabled=True,width="small"),
                    "__status": st.column_config.TextColumn("Status",disabled=True,width="medium"),
                }, use_container_width=True, hide_index=True, num_rows="fixed", key="al_de")
                if st.button("Save Corrections", key="al_sv"):
                    _ch = 0
                    for _,_row in _ed.iterrows():
                        _oi = int(_row["_row_id"])
                        _nl = _row["__predicted_label"]
                        if _oi in st.session_state.al_review_df.index:
                            _ol = st.session_state.al_review_df.at[_oi,"__predicted_label"]
                            st.session_state.al_review_df.at[_oi,"__predicted_label"] = _nl
                            if _nl != _ol:
                                st.session_state.al_review_df.at[_oi,"__status"] = "corrected"
                                _ch += 1
                    st.success(f"Saved! {_ch} label(s) corrected.")
                    st.rerun()
                with st.expander("Active Learning — Most Uncertain Samples"):
                    _un = _rdf.nsmallest(10,"__confidence")[[_tc3,"__predicted_label","__confidence","__status"]]
                    st.dataframe(_un,use_container_width=True)

        # ── TAB 4 EXPORT ─────────────────────────────────────────
        with al_tab4:
            if st.session_state.al_review_df is None:
                st.info("Complete auto-labeling first.")
            else:
                _exp = st.session_state.al_review_df.copy()
                _ea=(_exp["__status"]=="auto-accepted").sum()
                _ec=(_exp["__status"]=="corrected").sum()
                _ep=(_exp["__status"]=="needs review").sum()
                _eo=int(_exp["__source"].str.startswith("ollama").sum()) if "__source" in _exp.columns else 0
                _e1,_e2,_e3,_e4,_e5 = st.columns(5)
                for _col,_val,_lbl,_cls in [
                    (_e1,len(_exp),"Total",""),(_e2,_ea,"Accepted","green"),
                    (_e3,_ec,"Corrected",""),(_e4,_ep,"Pending","gold"),(_e5,_eo,"Ollama","purple")]:
                    _col.markdown(f'<div class="als-stat"><div class="als-stat-val {_cls}">{_val}</div>'
                                  f'<div class="als-stat-lbl">{_lbl}</div></div>',unsafe_allow_html=True)
                st.markdown('<div class="dm-divider"></div>',unsafe_allow_html=True)
                _ex1,_ex2 = st.columns(2)
                with _ex1:
                    _es = st.radio("Export scope",["All rows","Confirmed only"],key="al_es")
                with _ex2:
                    _lc = st.text_input("Label column name",value="label",key="al_lc")
                if "Confirmed" in _es: _exp = _exp[_exp["__status"]!="needs review"]
                _exp = _exp.rename(columns={"__predicted_label":_lc})
                _exp = _exp.drop(columns=["__status","__confidence","__source"],errors="ignore")
                st.download_button("Download Labeled CSV",
                    data=_exp.to_csv(index=False).encode("utf-8"),
                    file_name="datamind_labeled.csv",mime="text/csv",key="al_dl")
                st.dataframe(_exp.head(20),use_container_width=True)

    # ════════════════════════════════════════════════════════════
    #  IMAGE LABELING — Offline (cv2.dnn + YOLOv4-tiny)
    # ════════════════════════════════════════════════════════════
    else:
        _img_tab1, _img_tab2, _img_tab3 = st.tabs(
            ["⚙️  Setup & Upload", "🔍  Detect & Label", "📥  Export"])

        # ── COCO class names (80 classes) ────────────────────────────────
        _COCO_CLASSES = [
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
            "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
            "teddy bear","hair drier","toothbrush",
        ]

        # ── Model download helpers ────────────────────────────────────────
        _YOLO_MODELS = {
            "YOLOv4-tiny (23 MB — fastest)": {
                "cfg":     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                "size":    416,
                "key":     "yolov4-tiny",
            },
            "YOLOv3-tiny (34 MB — balanced)": {
                "cfg":     "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                "weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
                "size":    416,
                "key":     "yolov3-tiny",
            },
            "YOLOv3-full (237 MB — most accurate)": {
                "cfg":     "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                "weights": "https://pjreddie.com/media/files/yolov3.weights",
                "size":    608,
                "key":     "yolov3",
            },
        }

        import os as _os

        def _model_dir():
            d = _os.path.join(_os.path.expanduser("~"), ".datamind_models")
            _os.makedirs(d, exist_ok=True)
            return d

        def _download_file(url, dest_path, label=""):
            """Download with progress bar using requests (already in requirements)."""
            if _os.path.exists(dest_path):
                return True
            try:
                _prog_dl = st.progress(0, text=f"Downloading {label}...")
                resp = _rq.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            _prog_dl.progress(
                                min(downloaded / total, 1.0),
                                text=f"Downloading {label}: {downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB"
                            )
                _prog_dl.empty()
                return True
            except Exception as _de:
                st.error(f"Download failed: {_de}")
                if _os.path.exists(dest_path):
                    _os.remove(dest_path)
                return False

        def _load_yolo_net(model_info):
            """Download if needed and load YOLO via cv2.dnn. Cached in session state."""
            import cv2 as _cv2
            cache_key = "al_cv2_net_" + model_info["key"]
            if cache_key in st.session_state and st.session_state[cache_key] is not None:
                return st.session_state[cache_key]

            mdir    = _model_dir()
            cfg_p   = _os.path.join(mdir, model_info["key"] + ".cfg")
            wgt_p   = _os.path.join(mdir, model_info["key"] + ".weights")

            with st.status("Preparing YOLO model (one-time download)...", expanded=True) as _st:
                _st.write("Downloading config file...")
                if not _download_file(model_info["cfg"], cfg_p, "config"):
                    return None
                _st.write("Downloading weights...")
                if not _download_file(model_info["weights"], wgt_p, "weights"):
                    return None
                _st.write("Loading model into cv2.dnn...")
                net = _cv2.dnn.readNetFromDarknet(cfg_p, wgt_p)
                net.setPreferableBackend(_cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(_cv2.dnn.DNN_TARGET_CPU)
                st.session_state[cache_key] = net
                _st.update(label="Model ready!", state="complete")
            return net

        def _detect_objects(net, image_bytes, inp_size, conf_thresh, class_filter):
            """Run cv2.dnn YOLO detection. Returns list of box dicts + annotated PIL image."""
            import cv2 as _cv2
            import numpy as _np2
            from PIL import Image as _PILImg
            import io as _imgio

            # Decode image
            arr   = _np2.frombuffer(image_bytes, dtype=_np2.uint8)
            img   = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if img is None:
                _pil = _PILImg.open(_imgio.BytesIO(image_bytes)).convert("RGB")
                img  = _cv2.cvtColor(_np2.array(_pil), _cv2.COLOR_RGB2BGR)
            H, W  = img.shape[:2]

            # Build blob and forward pass
            blob = _cv2.dnn.blobFromImage(
                img, 1 / 255.0, (inp_size, inp_size),
                swapRB=True, crop=False
            )
            net.setInput(blob)
            layer_names  = net.getLayerNames()
            out_layers   = [layer_names[i - 1]
                            for i in net.getUnconnectedOutLayers().flatten()]
            outs = net.forward(out_layers)

            # Parse detections
            boxes_raw, confidences, class_ids = [], [], []
            for out in outs:
                for det in out:
                    scores   = det[5:]
                    cls_id   = int(_np2.argmax(scores))
                    conf     = float(scores[cls_id])
                    if conf < conf_thresh:
                        continue
                    cx, cy, bw, bh = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    boxes_raw.append([x1, y1, int(bw), int(bh)])
                    confidences.append(conf)
                    class_ids.append(cls_id)

            # NMS
            indices = _cv2.dnn.NMSBoxes(boxes_raw, confidences, conf_thresh, 0.4)
            indices = indices.flatten() if len(indices) else []

            boxes = []
            for idx in indices:
                x, y, bw, bh = boxes_raw[idx]
                cls_nm = _COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(_COCO_CLASSES) else "unknown"
                if class_filter and cls_nm not in class_filter:
                    continue
                boxes.append({
                    "class":      cls_nm,
                    "confidence": round(float(confidences[idx]), 3),
                    "x1": max(0, x),
                    "y1": max(0, y),
                    "x2": min(W, x + bw),
                    "y2": min(H, y + bh),
                })

            # Draw annotations on image
            ann_img = img.copy()
            colors  = {}
            for b in boxes:
                cls_nm = b["class"]
                if cls_nm not in colors:
                    _np2.random.seed(hash(cls_nm) % (2**32))
                    colors[cls_nm] = tuple(int(c) for c in _np2.random.randint(100, 255, 3))
                color = colors[cls_nm]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                _cv2.rectangle(ann_img, (x1, y1), (x2, y2), color, 2)
                label_text = f"{cls_nm} {b['confidence']:.2f}"
                (tw, th), _ = _cv2.getTextSize(label_text, _cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                _cv2.rectangle(ann_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                _cv2.putText(ann_img, label_text, (x1 + 2, y1 - 4),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            ann_rgb = _cv2.cvtColor(ann_img, _cv2.COLOR_BGR2RGB)
            ann_pil = _PILImg.fromarray(ann_rgb)
            ann_buf = _imgio.BytesIO()
            ann_pil.save(ann_buf, format="PNG")

            return boxes, ann_buf.getvalue()

        # ── IMG TAB 1 SETUP ──────────────────────────────────────────────
        with _img_tab1:
            st.markdown(
                '<div class="als-section">'
                '<div class="als-section-title">Model Selection</div>',
                unsafe_allow_html=True)

            st.markdown(
                '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;'
                'padding:.75rem 1rem;font-size:.82rem;color:#166534;margin-bottom:.8rem">'
                '&#9989; Fully offline — uses <strong>OpenCV cv2.dnn</strong> (already installed). '
                'No ultralytics, no torch, no API needed. '
                'Model weights download once (~23 MB) and are cached locally.</div>',
                unsafe_allow_html=True)

            _ym_name = st.selectbox(
                "YOLO model",
                list(_YOLO_MODELS.keys()),
                key="al_ym",
                help="YOLOv4-tiny is fastest and works great for most use cases."
            )
            st.session_state.al_yolo_model = _ym_name

            _yconf = st.slider(
                "Detection confidence threshold", 0.10, 0.95,
                st.session_state.al_img_conf, 0.05, key="al_yconf"
            )
            st.session_state.al_img_conf = _yconf
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="als-section">'
                '<div class="als-section-title">Upload Images</div>',
                unsafe_allow_html=True)
            _uploaded = st.file_uploader(
                "Upload images (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="al_img_up"
            )
            if _uploaded:
                st.session_state.al_img_files = _uploaded
                st.markdown(
                    f'<span class="dm-badge green">{len(_uploaded)} image(s) uploaded</span>',
                    unsafe_allow_html=True)
                _prev_cols = st.columns(min(3, len(_uploaded)))
                for _i, _img_f in enumerate(list(_uploaded)[:3]):
                    _prev_cols[_i].image(_img_f, caption=_img_f.name, use_container_width=True)
                if len(_uploaded) > 3:
                    st.caption(f"...and {len(_uploaded) - 3} more images")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="als-section">'
                '<div class="als-section-title">Class Filter (optional)</div>',
                unsafe_allow_html=True)
            st.caption(f"Leave empty to detect all 80 COCO classes. "
                       f"Available: {', '.join(_COCO_CLASSES[:10])}…")
            _clf_raw = st.text_input(
                "Filter to specific classes (comma-separated)",
                placeholder="e.g. person, car, dog",
                key="al_clf"
            )
            st.session_state.al_img_labels = (
                [c.strip() for c in _clf_raw.split(",") if c.strip()]
                if _clf_raw.strip() else []
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── IMG TAB 2 DETECT ─────────────────────────────────────────────
        with _img_tab2:
            if not st.session_state.al_img_files:
                st.markdown(
                    '<div style="text-align:center;padding:3rem;color:#94a3b8;">'
                    '<div style="font-size:2.5rem;margin-bottom:.5rem">&#128247;</div>'
                    '<div style="font-size:.9rem">Upload images in the Setup tab first</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                _nim = len(st.session_state.al_img_files)
                _mdl_info = _YOLO_MODELS[st.session_state.al_yolo_model]
                st.markdown(
                    f'<div style="background:#f0fdf4;border:1px solid #86efac;'
                    f'border-radius:10px;padding:.7rem 1rem;font-size:.8rem;'
                    f'color:#166534;margin-bottom:1rem;">'
                    f'{_nim} image(s) ready &middot; '
                    f'Model: <strong>{_mdl_info["key"]}</strong> &middot; '
                    f'Confidence: <strong>{st.session_state.al_img_conf:.0%}</strong> &middot; '
                    f'Input size: <strong>{_mdl_info["size"]}×{_mdl_info["size"]}</strong>'
                    f'</div>', unsafe_allow_html=True)

                if st.button("Run Detection (Offline)", key="al_yrun",
                             use_container_width=True, type="primary"):
                    try:
                        import cv2 as _cv2
                        _net = _load_yolo_net(_mdl_info)
                        if _net is None:
                            st.error("Failed to load model. Check your internet connection for the one-time download.")
                        else:
                            _results_list = []
                            _prog = st.progress(0, "Starting detection...")
                            for _ii, _img_file in enumerate(st.session_state.al_img_files):
                                _prog.progress(
                                    _ii / _nim,
                                    text=f"Detecting: {_img_file.name} ({_ii+1}/{_nim})"
                                )
                                _img_bytes = _img_file.read()
                                _boxes, _ann_bytes = _detect_objects(
                                    _net, _img_bytes,
                                    _mdl_info["size"],
                                    st.session_state.al_img_conf,
                                    st.session_state.al_img_labels,
                                )
                                _results_list.append({
                                    "filename":  _img_file.name,
                                    "boxes":     _boxes,
                                    "n_objects": len(_boxes),
                                    "classes":   list(set(b["class"] for b in _boxes)),
                                    "img_bytes": _img_bytes,
                                    "ann_bytes": _ann_bytes,
                                })
                                _prog.progress(
                                    (_ii + 1) / _nim,
                                    text=f"Done: {_img_file.name} ({_ii+1}/{_nim})"
                                )
                            _prog.empty()
                            st.session_state.al_img_results = _results_list
                            st.success(f"Detection complete on {_nim} image(s)!")

                    except ImportError:
                        st.error("OpenCV (cv2) not found. Run: pip install opencv-python-headless")
                    except Exception as _ye:
                        st.error(f"Detection error: {_ye}")
                        st.exception(_ye)

                # ── Results display ───────────────────────────────────────
                if st.session_state.al_img_results:
                    _res_list = st.session_state.al_img_results
                    _tot_obj  = sum(r["n_objects"] for r in _res_list)
                    _all_cls  = {}
                    for _r in _res_list:
                        for _b in _r["boxes"]:
                            _all_cls[_b["class"]] = _all_cls.get(_b["class"], 0) + 1

                    _ki1, _ki2, _ki3, _ki4 = st.columns(4)
                    for _col, _val, _lbl, _cls in [
                        (_ki1, _nim,             "Images",         ""),
                        (_ki2, _tot_obj,          "Objects Found",  "green"),
                        (_ki3, len(_all_cls),     "Unique Classes", "purple"),
                        (_ki4, f"{_tot_obj/_nim:.1f}" if _nim else 0, "Avg/Image", "gold"),
                    ]:
                        _col.markdown(
                            f'<div class="als-stat">'
                            f'<div class="als-stat-val {_cls}">{_val}</div>'
                            f'<div class="als-stat-lbl">{_lbl}</div></div>',
                            unsafe_allow_html=True)

                    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

                    if _all_cls:
                        _clsc1, _clsc2 = st.columns([1, 2])
                        with _clsc1:
                            st.markdown('<div class="als-section-title" '
                                        'style="color:#374151;font-size:.8rem;">Class Distribution</div>',
                                        unsafe_allow_html=True)
                            _fgc, _axc = plt.subplots(figsize=(4, max(2.5, len(_all_cls) * 0.4)))
                            _srt = dict(sorted(_all_cls.items(), key=lambda x: -x[1]))
                            _axc.barh(list(_srt.keys()), list(_srt.values()),
                                      color=C_BLUE, edgecolor="white")
                            _axc.set_xlabel("Count")
                            _fgc.tight_layout()
                            st.pyplot(_fgc)
                            plt.close(_fgc)
                        with _clsc2:
                            st.markdown('<div class="als-section-title" '
                                        'style="color:#374151;font-size:.8rem;">Summary</div>',
                                        unsafe_allow_html=True)
                            st.dataframe(
                                pd.DataFrame([
                                    {"Class": k, "Count": v,
                                     "Pct": f"{v * 100 // _tot_obj}%"}
                                    for k, v in _srt.items()
                                ]),
                                use_container_width=True, hide_index=True)

                    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="als-section-title" '
                                'style="color:#374151;font-size:.8rem;margin-bottom:.75rem;">'
                                'Per-Image Results</div>', unsafe_allow_html=True)
                    for _ri, _rr in enumerate(_res_list):
                        with st.expander(
                            f"{_rr['filename']} — {_rr['n_objects']} objects"
                            + (f" ({', '.join(_rr['classes'])})" if _rr['classes'] else " (none)"),
                            expanded=(_ri == 0)
                        ):
                            _rc1, _rc2 = st.columns(2)
                            with _rc1:
                                st.markdown("**Original**")
                                st.image(_rr["img_bytes"], use_container_width=True)
                            with _rc2:
                                st.markdown("**Annotated**")
                                st.image(_rr["ann_bytes"], use_container_width=True)
                            if _rr["boxes"]:
                                st.dataframe(
                                    pd.DataFrame(_rr["boxes"])[
                                        ["class", "confidence", "x1", "y1", "x2", "y2"]],
                                    use_container_width=True, hide_index=True)
                            else:
                                st.caption("No objects detected in this image.")

        # ── IMG TAB 3 EXPORT ─────────────────────────────────────────────
        with _img_tab3:
            if not st.session_state.al_img_results:
                st.info("Run detection first (Detect & Label tab).")
            else:
                _exp_list = st.session_state.al_img_results
                _rows = []
                for _rr in _exp_list:
                    if _rr["boxes"]:
                        for _bb in _rr["boxes"]:
                            _rows.append({
                                "filename":   _rr["filename"],
                                "class":      _bb["class"],
                                "confidence": _bb["confidence"],
                                "x1": _bb["x1"], "y1": _bb["y1"],
                                "x2": _bb["x2"], "y2": _bb["y2"],
                            })
                    else:
                        _rows.append({
                            "filename": _rr["filename"], "class": "", "confidence": 0,
                            "x1": 0, "y1": 0, "x2": 0, "y2": 0
                        })

                _df_exp = pd.DataFrame(_rows)
                _ei1, _ei2, _ei3 = st.columns(3)
                _ei1.markdown(
                    f'<div class="als-stat"><div class="als-stat-val">{len(_exp_list)}</div>'
                    f'<div class="als-stat-lbl">Images</div></div>', unsafe_allow_html=True)
                _ei2.markdown(
                    f'<div class="als-stat"><div class="als-stat-val green">'
                    f'{sum(r["n_objects"] for r in _exp_list)}</div>'
                    f'<div class="als-stat-lbl">Total Objects</div></div>', unsafe_allow_html=True)
                _ei3.markdown(
                    f'<div class="als-stat"><div class="als-stat-val purple">'
                    f'{_df_exp["class"].nunique()}</div>'
                    f'<div class="als-stat-lbl">Classes</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
                st.download_button(
                    "Download Labels CSV",
                    data=_df_exp.to_csv(index=False).encode("utf-8"),
                    file_name="datamind_image_labels.csv",
                    mime="text/csv", key="al_img_dl")
                st.dataframe(_df_exp, use_container_width=True, hide_index=True)

                st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
                st.markdown("**YOLO .txt annotation format preview:**")
                _yolo_txt = ""
                for _rr in _exp_list[:2]:
                    _yolo_txt += f"# {_rr['filename']}\n"
                    for _bb in _rr["boxes"][:5]:
                        _yolo_txt += (
                            f"{_COCO_CLASSES.index(_bb['class']) if _bb['class'] in _COCO_CLASSES else 0} "
                            f"{_bb['x1']} {_bb['y1']} "
                            f"{_bb['x2']} {_bb['y2']} "
                            f"{_bb['confidence']:.3f}\n"
                        )
                st.code(_yolo_txt, language="text")

elif page == "🛡️ Admin Panel":
    # ── Only accessible if admin=True ────────────────────────────────────
    if not st.session_state.auth_is_admin:
        st.error("Access denied. Admin only.")
        st.stop()

    st.markdown(
        '<div class="dm-pagehead"><div class="icon">🛡️</div>'
        '<div><div class="title">Admin Panel</div>'
        '<div class="sub">Manage users, premium status and payments</div>'
        '</div></div>', unsafe_allow_html=True)

    if not RTDB_OK:
        st.error(
            "Firebase Realtime Database is not connected. "
            "Fill in _FB_CONFIG and check your databaseURL.")
        st.stop()

    # ── Refresh button ────────────────────────────────────────────────────
    _col_refresh, _ = st.columns([1, 4])
    with _col_refresh:
        if st.button("Refresh Users", key="admin_refresh"):
            st.rerun()

    # ── Load all users ────────────────────────────────────────────────────
    _all_users = _rtdb_get_all_users()

    if not _all_users:
        st.info("No users found in the Realtime Database yet.")
        st.stop()

    # ── Summary KPIs ──────────────────────────────────────────────────────
    _total   = len(_all_users)
    _premium = sum(1 for u in _all_users if u.get("plan") == "premium")
    _pending = sum(1 for u in _all_users if u.get("plan") == "pending_review")
    _free    = _total - _premium - _pending
    _admins  = sum(1 for u in _all_users if u.get("admin", False))

    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    _k1.markdown(
        '<div class="dm-kpi"><div class="val">' + str(_total) + '</div>'
        '<div class="lbl">Total Users</div></div>', unsafe_allow_html=True)
    _k2.markdown(
        '<div class="dm-kpi green"><div class="val">' + str(_premium) + '</div>'
        '<div class="lbl">Premium</div></div>', unsafe_allow_html=True)
    _k3.markdown(
        '<div class="dm-kpi"><div class="val">' + str(_free) + '</div>'
        '<div class="lbl">Free</div></div>', unsafe_allow_html=True)
    _k4.markdown(
        '<div class="dm-kpi gold"><div class="val">' + str(_admins) + '</div>'
        '<div class="lbl">Admins</div></div>', unsafe_allow_html=True)
    _k5.markdown(
        '<div class="dm-kpi" style="border-color:#fde68a;">'
        '<div class="val" style="color:#d97706;">' + str(_pending) + '</div>'
        '<div class="lbl">⏳ Pending Review</div></div>', unsafe_allow_html=True)

    # ── Pending payment approvals ─────────────────────────────────────────
    _pending_users = [u for u in _all_users if u.get("plan") == "pending_review"]
    if _pending_users:
        st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dm-card-title" style="color:#d97706;">⏳ Pending Payment Approvals ({len(_pending_users)})</div>',
            unsafe_allow_html=True)
        st.warning(
            "These users have submitted a UTR for verification. "
            "Check your UPI app / bank statement before approving.")
        for _pu in _pending_users:
            _p_uid   = _pu.get("_uid", "")
            _p_email = _pu.get("email", "unknown")
            _p_utr   = _pu.get("last_utr", "—")
            _p_ts    = _pu.get("payment_ts", "")[:16].replace("T", " ")
            with st.expander(f"🟡 {_p_email}  ·  UTR: {_p_utr}  ·  Submitted: {_p_ts}"):
                st.markdown(
                    f'<div style="font-size:.8rem;margin-bottom:.75rem;">'
                    f'<b>UID:</b> <code>{_p_uid}</code>&nbsp;&nbsp;'
                    f'<b>UTR:</b> <code>{_p_utr}</code>&nbsp;&nbsp;'
                    f'<b>Submitted:</b> {_p_ts}</div>',
                    unsafe_allow_html=True)
                _pa1, _pa2 = st.columns(2)
                with _pa1:
                    if st.button(f"✅ Approve — Grant Premium", key="approve_" + _p_uid):
                        _paid_until = (
                            datetime.datetime.utcnow() +
                            datetime.timedelta(days=30)).isoformat()
                        _rtdb_update_user(_p_uid, {
                            "plan":       "premium",
                            "paid_until": _paid_until,
                            "proj_used":  0,
                        })
                        st.success(f"Premium granted to {_p_email}.")
                        st.rerun()
                with _pa2:
                    if st.button(f"❌ Reject — Move to Free", key="reject_" + _p_uid):
                        _rtdb_update_user(_p_uid, {
                            "plan":     "free",
                            "last_utr": "",
                        })
                        st.warning(f"{_p_email} moved back to Free.")
                        st.rerun()
    else:
        st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
        st.success("✅ No pending payment approvals.")
    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

    # ── User table + individual controls ─────────────────────────────────
    st.markdown('<div class="dm-card-title">All Users</div>',
                unsafe_allow_html=True)

    for _u in _all_users:
        _uid   = _u.get("_uid", "")
        _email = _u.get("email", "unknown")
        _plan  = _u.get("plan", "free")
        _admin = bool(_u.get("admin", False))
        _used  = _u.get("proj_used", 0)
        _pu    = _u.get("paid_until", None)
        _utr   = _u.get("last_utr", "")
        _login = _u.get("last_login", "")[:16].replace("T", " ")
        _join  = _u.get("created_at", "")[:10]

        _plan_color  = "#047857" if _plan == "premium" else "#6b7280"
        _admin_badge = (
            '<span style="background:#fef3c7;color:#b45309;border-radius:20px;'
            'padding:2px 8px;font-size:.65rem;font-weight:700;margin-left:6px;">'
            'ADMIN</span>' if _admin else "")

        with st.expander(
                _email + (" [ADMIN]" if _admin else "") +
                " — " + _plan.upper(), expanded=False):

            _ic1, _ic2, _ic3 = st.columns(3)
            _ic1.markdown(
                '<div class="dm-kpi"><div class="val" style="font-size:1rem;color:'
                + _plan_color + '">' + _plan.capitalize() + '</div>'
                '<div class="lbl">Plan</div></div>', unsafe_allow_html=True)
            _ic2.markdown(
                '<div class="dm-kpi"><div class="val" style="font-size:1rem">'
                + str(_used) + '</div>'
                '<div class="lbl">Projects Used</div></div>',
                unsafe_allow_html=True)
            _ic3.markdown(
                '<div class="dm-kpi"><div class="val" style="font-size:.85rem">'
                + str(_login or "—") + '</div>'
                '<div class="lbl">Last Login</div></div>',
                unsafe_allow_html=True)

            st.markdown(
                '<div style="font-size:.75rem;color:#6b7280;margin:.5rem 0">'
                'UID: <code>' + _uid + '</code>'
                ' &nbsp;|&nbsp; Joined: ' + str(_join) +
                (' &nbsp;|&nbsp; Last UTR: <code>' + _utr + '</code>' if _utr else '') +
                (' &nbsp;|&nbsp; Paid until: ' + str(_pu)[:10] if _pu else '') +
                '</div>', unsafe_allow_html=True)

            st.markdown("**Actions:**")
            _ac1, _ac2, _ac3, _ac4 = st.columns(4)

            with _ac1:
                if _plan != "premium":
                    if st.button("Grant Premium", key="gp_" + _uid):
                        _paid_until = (
                            datetime.datetime.utcnow() +
                            datetime.timedelta(days=30)).isoformat()
                        _rtdb_update_user(_uid, {
                            "plan": "premium",
                            "paid_until": _paid_until,
                            "proj_used": 0,
                        })
                        st.success("Premium granted for 30 days.")
                        st.rerun()
                    if _plan == "pending_review":
                        if st.button("Reject Payment", key="rjct_" + _uid):
                            _rtdb_update_user(_uid, {"plan": "free", "last_utr": ""})
                            st.warning("Payment rejected. User moved to Free.")
                            st.rerun()
                else:
                    if st.button("Revoke Premium", key="rp_" + _uid):
                        _rtdb_update_user(_uid, {
                            "plan": "free",
                            "paid_until": None,
                        })
                        st.warning("Premium revoked.")
                        st.rerun()

            with _ac2:
                _days_ext = st.number_input(
                    "Extend (days)", min_value=1, max_value=365,
                    value=30, key="ext_days_" + _uid, label_visibility="collapsed")
                if st.button("Extend " + str(int(_days_ext)) + "d",
                             key="ext_" + _uid):
                    _base = datetime.datetime.utcnow()
                    if _pu:
                        try:
                            _base = max(_base,
                                datetime.datetime.fromisoformat(str(_pu)))
                        except Exception:
                            pass
                    _new_pu = (_base + datetime.timedelta(
                        days=int(_days_ext))).isoformat()
                    _rtdb_update_user(_uid, {
                        "plan": "premium",
                        "paid_until": _new_pu,
                    })
                    st.success("Extended until " + _new_pu[:10])
                    st.rerun()

            with _ac3:
                if not _admin:
                    if st.button("Make Admin", key="ma_" + _uid):
                        _rtdb_update_user(_uid, {"admin": True})
                        st.success("Admin granted.")
                        st.rerun()
                else:
                    if _uid != st.session_state.auth_uid:
                        if st.button("Remove Admin", key="ra_" + _uid):
                            _rtdb_update_user(_uid, {"admin": False})
                            st.warning("Admin removed.")
                            st.rerun()
                    else:
                        st.caption("(You)")

            with _ac4:
                if st.button("Reset Projects", key="rpr_" + _uid):
                    _rtdb_update_user(_uid, {"proj_used": 0})
                    st.success("Project counter reset.")
                    st.rerun()

    # ── Export all users as CSV ───────────────────────────────────────────
    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-card-title">Export</div>',
                unsafe_allow_html=True)
    if _all_users:
        import io as _admin_io
        _rows = []
        for _u in _all_users:
            _rows.append({
                "uid":        _u.get("_uid", ""),
                "email":      _u.get("email", ""),
                "plan":       _u.get("plan", "free"),
                "admin":      _u.get("admin", False),
                "proj_used":  _u.get("proj_used", 0),
                "paid_until": _u.get("paid_until", ""),
                "last_login": _u.get("last_login", ""),
                "created_at": _u.get("created_at", ""),
                "last_utr":   _u.get("last_utr", ""),
            })
        _df_export = pd.DataFrame(_rows)
        st.dataframe(_df_export, use_container_width=True)
        st.download_button(
            "Download Users CSV",
            data=_df_export.to_csv(index=False).encode("utf-8"),
            file_name="datamind_users.csv",
            mime="text/csv",
            key="admin_export_csv",
        )
