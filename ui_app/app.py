"""
Streamlit UI Application - DeepText Multi-Task Learning Demo
Giao diện web để tương tác với model từ checkpoint
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Thêm path để import model_service
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_service import get_model_service, ModelService

# Page config
st.set_page_config(
    page_title="DeepText Multi-Task Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern & Beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Cards & Boxes */
    .prediction-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 0.75rem;
        border-left: 4px solid;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 1.5rem;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_service' not in st.session_state:
    st.session_state.model_service = None
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False

def auto_load_model():
    """Tự động load model khi app khởi động"""
    if st.session_state.model_loaded or st.session_state.auto_load_attempted:
        return
    
    # Đường dẫn mặc định (từ root của repo trên Streamlit Cloud)
    default_paths = {
        'model': [
            'checkpoints/models/best_model_20251027_085402.h5',
            'DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5',
        ],
        'config': [
            'config_default.json',
            'DeepText-MTL/config_default.json',
        ],
        'train_data': [
            'checkpoints/train_clean.csv',
            'DeepText-MTL/checkpoints/train_clean.csv',
        ]
    }
    
    # Tìm đường dẫn tồn tại
    model_path = None
    config_path = None
    train_data_path = None
    
    for path in default_paths['model']:
        if os.path.exists(path):
            model_path = path
            break
    
    for path in default_paths['config']:
        if os.path.exists(path):
            config_path = path
            break
    
    for path in default_paths['train_data']:
        if os.path.exists(path):
            train_data_path = path
            break
    
    # Nếu tìm thấy đủ model và training data, tự động load
    if model_path and train_data_path:
        try:
            with st.spinner("🔄 Đang tự động load model... Vui lòng đợi..."):
                st.session_state.model_service = get_model_service(
                    model_path=model_path,
                    config_path=config_path,
                    train_data_path=train_data_path
                )
                st.session_state.model_loaded = True
                st.session_state.auto_load_attempted = True
                st.rerun()
        except Exception as e:
            st.session_state.auto_load_attempted = True
            # Không hiển thị lỗi, để user tự load nếu cần
    
def load_model():
    """Load model vào session state"""
    # Đường dẫn mặc định (tự động detect)
    default_model_paths = [
        'checkpoints/models/best_model_20251027_085402.h5',
        'DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5'
    ]
    
    default_config_paths = [
        'config_default.json',
        'DeepText-MTL/config_default.json'
    ]
    
    default_train_paths = [
        'checkpoints/train_clean.csv',
        'DeepText-MTL/checkpoints/train_clean.csv'
    ]
    
    # Tìm đường dẫn tồn tại
    default_model = next((p for p in default_model_paths if os.path.exists(p)), default_model_paths[0])
    default_config = next((p for p in default_config_paths if os.path.exists(p)), default_config_paths[0])
    default_train = next((p for p in default_train_paths if os.path.exists(p)), default_train_paths[0])
    
    model_path = st.sidebar.text_input(
        "Đường dẫn Model",
        value=default_model
    )
    
    config_path = st.sidebar.text_input(
        "Đường dẫn Config (optional)",
        value=default_config
    )
    
    train_data_path = st.sidebar.text_input(
        "Đường dẫn Training Data (để fit tokenizer)",
        value=default_train
    )
    
    if st.sidebar.button("🔄 Load Model", type="primary"):
        # Validate model path
        if not model_path or not model_path.strip():
            st.sidebar.error("❌ Vui lòng nhập đường dẫn model!")
            return
        
        if not os.path.exists(model_path):
            st.sidebar.error(f"❌ File model không tồn tại: {model_path}")
            st.sidebar.info("💡 Kiểm tra lại đường dẫn hoặc sử dụng đường dẫn tuyệt đối")
            return
        
        # Validate config path (optional)
        if config_path and config_path.strip() and not os.path.exists(config_path):
            st.sidebar.warning(f"⚠️ File config không tồn tại: {config_path}")
            st.sidebar.info("💡 Sẽ tiếp tục không dùng config")
            config_path = None
        
        # Validate training data path
        if not train_data_path or not train_data_path.strip():
            st.sidebar.error("❌ Training data path là bắt buộc để fit tokenizer!")
            st.sidebar.info("💡 Tokenizer cần được fit từ training data để vocabulary khớp với model")
            return
        elif not os.path.exists(train_data_path):
            st.sidebar.error(f"❌ File training data không tồn tại: {train_data_path}")
            st.sidebar.info("💡 Vui lòng kiểm tra lại đường dẫn hoặc sử dụng đường dẫn tuyệt đối")
            return
        
        # Load model
        try:
            with st.spinner("Đang load model... Vui lòng đợi..."):
                st.session_state.model_service = get_model_service(
                    model_path=model_path,
                    config_path=config_path,
                    train_data_path=train_data_path
                )
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model đã được load thành công!")
                st.rerun()
                
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi load model: {str(e)}")
            st.session_state.model_loaded = False
            st.session_state.model_service = None

def main():
    """Main application"""
    
    # Tự động load model khi khởi động (nếu chưa load)
    auto_load_model()
    
    # Beautiful Header với gradient
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🤖 DeepText Multi-Task Learning</h1>
        <p class="sub-header">Phân tích cảm xúc, phát hiện ngôn từ thù địch và bạo lực</p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 2rem; font-size: 0.9rem; font-weight: 600;">🎭 Emotion</span>
            <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 2rem; font-size: 0.9rem; font-weight: 600;">😡 Hate Speech</span>
            <span style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 2rem; font-size: 0.9rem; font-weight: 600;">⚔️ Violence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Hiển thị thông tin nếu đã auto-load
        if st.session_state.model_loaded and st.session_state.auto_load_attempted:
            st.success("✅ Model đã tự động load!")
            st.caption("💡 Nếu cần load model khác, click Reset và nhập đường dẫn mới")
        else:
            load_model()
        
        if st.session_state.model_loaded:
            st.markdown("---")
            st.success("✅ Model đã sẵn sàng")
            
            # Model info
            if st.button("ℹ️ Thông tin Model"):
                info = st.session_state.model_service.get_model_info()
                st.json(info)
            
            # Reset button
            if st.button("🔄 Reset Model", type="secondary"):
                st.session_state.model_loaded = False
                st.session_state.model_service = None
                # Reset singleton
                from ui_app.model_service import reset_model_service
                reset_model_service()
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.warning("⚠️ Vui lòng load model từ sidebar để bắt đầu sử dụng.")
        st.info("""
        **Hướng dẫn:**
        1. Nhập đường dẫn đến file model (.h5) trong sidebar
        2. Nhập đường dẫn đến file config (optional)
        3. Nhập đường dẫn đến training data để fit tokenizer
        4. Click "Load Model" để khởi tạo
        """)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "📈 Visualizations", "ℹ️ About"])
    
    with tab1:
        st.header("Phân tích Text đơn lẻ")
        
        # Input text
        text_input = st.text_area(
            "Nhập text cần phân tích:",
            height=150,
            placeholder="Ví dụ: Tôi cảm thấy rất vui vẻ và hạnh phúc hôm nay!"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🔍 Phân tích", type="primary", use_container_width=True)
        
        if predict_button and text_input.strip():
            with st.spinner("Đang xử lý..."):
                try:
                    prediction = st.session_state.model_service.predict(text_input.strip())
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Kết quả phân tích")
                    
                    # Emotion
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🎭 Cảm xúc")
                        emotion_label = prediction['emotion']['label']
                        emotion_conf = prediction['emotion']['confidence']
                        
                        # Map emotion to emoji
                        emotion_emoji = {
                            'sad': '😢',
                            'joy': '😊',
                            'love': '❤️',
                            'angry': '😠',
                            'fear': '😨',
                            'surprise': '😲',
                            'no_emo': '😐'
                        }
                        
                        emoji = emotion_emoji.get(emotion_label, '😐')
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem; color: white;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                            <div style="font-size: 1.2rem; font-weight: 600;">{emotion_label.capitalize()}</div>
                            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">Độ tin cậy: {emotion_conf:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Emotion probabilities chart với gradient đẹp
                    emotion_df = pd.DataFrame({
                        'Emotion': list(prediction['emotion']['probabilities'].keys()),
                        'Probability': list(prediction['emotion']['probabilities'].values())
                    })
                    fig_emotion = px.bar(
                        emotion_df,
                        x='Emotion',
                        y='Probability',
                        title='📊 Phân bố Cảm xúc',
                        color='Probability',
                        color_continuous_scale=px.colors.sequential.Purples,
                        text='Probability'
                    )
                    fig_emotion.update_traces(
                        texttemplate='%{text:.1%}',
                        textposition='outside',
                        marker=dict(
                            line=dict(color='rgba(0,0,0,0.1)', width=1),
                            cornerradius=8
                        )
                    )
                    fig_emotion.update_layout(
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter", size=12),
                        title_font=dict(size=18, color='#1e293b'),
                        xaxis=dict(title='', tickfont=dict(size=11)),
                        yaxis=dict(title='Xác suất', tickformat='.0%')
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 😡 Ngôn từ thù địch")
                        hate_labels = prediction['hate']['labels']
                        hate_confidences = prediction['hate']['confidences']
                        
                        if hate_labels:
                            for label in hate_labels:
                                conf = hate_confidences[label]
                                st.metric(
                                    label.capitalize(),
                                    f"{conf:.2%}"
                                )
                        else:
                            st.success("✅ Không phát hiện ngôn từ thù địch")
                        
                        # Hate probabilities chart với gradient đẹp
                        hate_df = pd.DataFrame({
                            'Category': list(prediction['hate']['probabilities'].keys()),
                            'Probability': list(prediction['hate']['probabilities'].values())
                        })
                        fig_hate = px.bar(
                            hate_df,
                            x='Category',
                            y='Probability',
                            title='📊 Phân bố Ngôn từ thù địch',
                            color='Probability',
                            color_continuous_scale=px.colors.sequential.Reds,
                            text='Probability'
                        )
                        fig_hate.update_traces(
                            texttemplate='%{text:.1%}',
                            textposition='outside',
                            marker=dict(
                                line=dict(color='rgba(0,0,0,0.1)', width=1),
                                cornerradius=8
                            )
                        )
                        fig_hate.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=12),
                            title_font=dict(size=18, color='#1e293b'),
                            xaxis=dict(title='', tickfont=dict(size=11)),
                            yaxis=dict(title='Xác suất', tickformat='.0%')
                        )
                        st.plotly_chart(fig_hate, use_container_width=True)
                    
                    with col3:
                        st.markdown("### ⚔️ Bạo lực")
                        violence_labels = prediction['violence']['labels']
                        violence_confidences = prediction['violence']['confidences']
                        
                        if violence_labels:
                            for label in violence_labels:
                                conf = violence_confidences[label]
                                st.metric(
                                    label.capitalize(),
                                    f"{conf:.2%}"
                                )
                        else:
                            st.success("✅ Không phát hiện nội dung bạo lực")
                        
                        # Violence probabilities chart với gradient đẹp
                        violence_df = pd.DataFrame({
                            'Category': list(prediction['violence']['probabilities'].keys()),
                            'Probability': list(prediction['violence']['probabilities'].values())
                        })
                        fig_violence = px.bar(
                            violence_df,
                            x='Category',
                            y='Probability',
                            title='📊 Phân bố Bạo lực',
                            color='Probability',
                            color_continuous_scale=px.colors.sequential.Oranges,
                            text='Probability'
                        )
                        fig_violence.update_traces(
                            texttemplate='%{text:.1%}',
                            textposition='outside',
                            marker=dict(
                                line=dict(color='rgba(0,0,0,0.1)', width=1),
                                cornerradius=8
                            )
                        )
                        fig_violence.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=12),
                            title_font=dict(size=18, color='#1e293b'),
                            xaxis=dict(title='', tickfont=dict(size=11)),
                            yaxis=dict(title='Xác suất', tickformat='.0%')
                        )
                        st.plotly_chart(fig_violence, use_container_width=True)
                    
                    # Combined visualization với gradient đẹp
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="font-size: 1.8rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">📈 Tổng quan Predictions</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create combined chart với màu gradient đẹp
                    fig_combined = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('🎭 Cảm xúc', '😡 Ngôn từ thù địch', '⚔️ Bạo lực'),
                        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
                        horizontal_spacing=0.1
                    )
                    
                    # Emotion với gradient purple
                    emotion_colors = px.colors.sequential.Purples
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['emotion']['probabilities'].keys()),
                            y=list(prediction['emotion']['probabilities'].values()),
                            name='Cảm xúc',
                            marker=dict(
                                color=list(prediction['emotion']['probabilities'].values()),
                                colorscale='Purples',
                                line=dict(color='rgba(0,0,0,0.1)', width=1),
                                cornerradius=8
                            ),
                            text=[f"{v:.1%}" for v in prediction['emotion']['probabilities'].values()],
                            textposition='outside'
                        ),
                        row=1, col=1
                    )
                    
                    # Hate với gradient red
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['hate']['probabilities'].keys()),
                            y=list(prediction['hate']['probabilities'].values()),
                            name='Ngôn từ thù địch',
                            marker=dict(
                                color=list(prediction['hate']['probabilities'].values()),
                                colorscale='Reds',
                                line=dict(color='rgba(0,0,0,0.1)', width=1),
                                cornerradius=8
                            ),
                            text=[f"{v:.1%}" for v in prediction['hate']['probabilities'].values()],
                            textposition='outside'
                        ),
                        row=1, col=2
                    )
                    
                    # Violence với gradient orange
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['violence']['probabilities'].keys()),
                            y=list(prediction['violence']['probabilities'].values()),
                            name='Bạo lực',
                            marker=dict(
                                color=list(prediction['violence']['probabilities'].values()),
                                colorscale='Oranges',
                                line=dict(color='rgba(0,0,0,0.1)', width=1),
                                cornerradius=8
                            ),
                            text=[f"{v:.1%}" for v in prediction['violence']['probabilities'].values()],
                            textposition='outside'
                        ),
                        row=1, col=3
                    )
                    
                    fig_combined.update_layout(
                        height=450,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter", size=11),
                        title_font=dict(size=20, color='#1e293b'),
                        margin=dict(l=20, r=20, t=60, b=40)
                    )
                    
                    # Update x-axis labels
                    for i in range(1, 4):
                        fig_combined.update_xaxes(tickangle=-45, row=1, col=i)
                        fig_combined.update_yaxes(tickformat='.0%', row=1, col=i)
                    
                    st.plotly_chart(fig_combined, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi predict: {str(e)}")
        elif predict_button:
            st.warning("⚠️ Vui lòng nhập text để phân tích")
    
    with tab2:
        st.header("Phân tích Batch (nhiều text)")
        
        # Option 1: Upload CSV
        uploaded_file = st.file_uploader(
            "Upload file CSV (cột 'text' chứa các text cần phân tích)",
            type=['csv']
        )
        
        # Option 2: Input multiple texts
        st.markdown("**Hoặc nhập nhiều text:**")
        batch_texts = st.text_area(
            "Nhập nhiều text (mỗi dòng một text):",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            batch_predict_button = st.button("🔍 Phân tích Batch", type="primary", use_container_width=True)
        
        if batch_predict_button:
            texts_to_process = []
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_process = df['text'].astype(str).tolist()
                    else:
                        st.error("❌ File CSV phải có cột 'text'")
                        return
                except Exception as e:
                    st.error(f"❌ Lỗi khi đọc file: {str(e)}")
                    return
            elif batch_texts.strip():
                texts_to_process = [t.strip() for t in batch_texts.split('\n') if t.strip()]
            else:
                st.warning("⚠️ Vui lòng upload file hoặc nhập text")
                return
            
            if texts_to_process:
                with st.spinner(f"Đang xử lý {len(texts_to_process)} texts..."):
                    try:
                        results = st.session_state.model_service.predict_batch(texts_to_process)
                        
                        # Create results DataFrame
                        results_data = []
                        for r in results:
                            results_data.append({
                                'Text': r['text'],
                                'Emotion': r['emotion']['label'],
                                'Emotion Confidence': f"{r['emotion']['confidence']:.2%}",
                                'Hate Labels': ', '.join(r['hate']['labels']) if r['hate']['labels'] else 'None',
                                'Violence Labels': ', '.join(r['violence']['labels']) if r['violence']['labels'] else 'None'
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        
                        st.success(f"✅ Đã phân tích {len(results)} texts")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Download kết quả CSV",
                            data=csv,
                            file_name="predictions_batch.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi khi predict batch: {str(e)}")
    
    with tab3:
        st.header("Visualizations & Analytics")
        
        if st.session_state.model_loaded and st.session_state.model_service:
            try:
                model_info = st.session_state.model_service.get_model_info()
                
                st.subheader("📊 Model Information")
                st.json(model_info)
                
                st.subheader("📈 Task Configuration")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎭 Emotion Classes**")
                    emotion_classes = model_info.get('emotion_classes', ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'])
                    for i, cls in enumerate(emotion_classes, 1):
                        st.write(f"{i}. {cls}")
                
                with col2:
                    st.markdown("**😡 Hate Classes**")
                    hate_classes = model_info.get('hate_classes', ['hate', 'offensive', 'neutral'])
                    for i, cls in enumerate(hate_classes, 1):
                        st.write(f"{i}. {cls}")
                
                with col3:
                    st.markdown("**⚔️ Violence Classes**")
                    violence_classes = model_info.get('violence_classes', ['sex_viol', 'phys_viol', 'no_viol'])
                    for i, cls in enumerate(violence_classes, 1):
                        st.write(f"{i}. {cls}")
            except Exception as e:
                st.error(f"❌ Lỗi khi lấy thông tin model: {str(e)}")
                st.info("💡 Vui lòng thử load lại model")
        else:
            st.info("Vui lòng load model để xem thông tin")
    
    with tab4:
        st.header("ℹ️ About")
        
        st.markdown("""
        ## DeepText Multi-Task Learning
        
        Ứng dụng này tích hợp với checkpoint models của DeepText Multi-Task Learning model để:
        
        ### 🎯 Chức năng chính:
        
        1. **🎭 Phân tích cảm xúc** (7 classes)
           - Sad, Joy, Love, Angry, Fear, Surprise, No Emotion
        
        2. **😡 Phát hiện ngôn từ thù địch** (3 classes)
           - Hate, Offensive, Neutral
        
        3. **⚔️ Phát hiện bạo lực** (3 classes)
           - Sexual Violence, Physical Violence, No Violence
        
        ### 📋 Workflow:
        
        1. **Load Model**: Load model từ checkpoint (.h5 file)
        2. **Load Tokenizer**: Fit tokenizer từ training data
        3. **Preprocess**: Convert text thành sequences và padding
        4. **Predict**: Model inference với 3 outputs
        5. **Post-process**: Interpret predictions và hiển thị kết quả
        
        ### 🔧 Technical Stack:
        
        - **Backend**: TensorFlow/Keras
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        
        ### 📖 Usage:
        
        1. Chạy app: `streamlit run app.py`
        2. Load model từ sidebar
        3. Nhập text và nhận predictions
        4. Xem visualizations và export kết quả
        
        ### 🚀 Features:
        
        - ✅ Single text prediction
        - ✅ Batch prediction từ CSV
        - ✅ Interactive visualizations
        - ✅ Export results
        - ✅ Model information display
        """)

if __name__ == "__main__":
    main()

