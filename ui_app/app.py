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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_service' not in st.session_state:
    st.session_state.model_service = None

def load_model():
    """Load model vào session state"""
    model_path = st.sidebar.text_input(
        "Đường dẫn Model",
        value="DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5"
    )
    
    config_path = st.sidebar.text_input(
        "Đường dẫn Config (optional)",
        value="DeepText-MTL/config_default.json"
    )
    
    train_data_path = st.sidebar.text_input(
        "Đường dẫn Training Data (để fit tokenizer)",
        value="DeepText-MTL/checkpoints/train_clean.csv"
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
    
    # Header
    st.markdown('<h1 class="main-header">🤖 DeepText Multi-Task Learning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân tích cảm xúc, phát hiện ngôn từ thù địch và bạo lực</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
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
                        st.metric(
                            "Nhãn",
                            f"{emoji} {emotion_label.capitalize()}"
                        )
                        st.metric(
                            "Độ tin cậy",
                            f"{emotion_conf:.2%}"
                        )
                        
                        # Emotion probabilities chart
                        emotion_df = pd.DataFrame({
                            'Emotion': list(prediction['emotion']['probabilities'].keys()),
                            'Probability': list(prediction['emotion']['probabilities'].values())
                        })
                        fig_emotion = px.bar(
                            emotion_df,
                            x='Emotion',
                            y='Probability',
                            title='Phân bố Cảm xúc',
                            color='Probability',
                            color_continuous_scale='Blues'
                        )
                        fig_emotion.update_layout(height=300)
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
                        
                        # Hate probabilities chart
                        hate_df = pd.DataFrame({
                            'Category': list(prediction['hate']['probabilities'].keys()),
                            'Probability': list(prediction['hate']['probabilities'].values())
                        })
                        fig_hate = px.bar(
                            hate_df,
                            x='Category',
                            y='Probability',
                            title='Phân bố Ngôn từ thù địch',
                            color='Probability',
                            color_continuous_scale='Reds'
                        )
                        fig_hate.update_layout(height=300)
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
                        
                        # Violence probabilities chart
                        violence_df = pd.DataFrame({
                            'Category': list(prediction['violence']['probabilities'].keys()),
                            'Probability': list(prediction['violence']['probabilities'].values())
                        })
                        fig_violence = px.bar(
                            violence_df,
                            x='Category',
                            y='Probability',
                            title='Phân bố Bạo lực',
                            color='Probability',
                            color_continuous_scale='Oranges'
                        )
                        fig_violence.update_layout(height=300)
                        st.plotly_chart(fig_violence, use_container_width=True)
                    
                    # Combined visualization
                    st.markdown("---")
                    st.subheader("📈 Tổng quan Predictions")
                    
                    # Create combined chart
                    fig_combined = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('Cảm xúc', 'Ngôn từ thù địch', 'Bạo lực'),
                        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    # Emotion
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['emotion']['probabilities'].keys()),
                            y=list(prediction['emotion']['probabilities'].values()),
                            name='Cảm xúc',
                            marker_color='blue'
                        ),
                        row=1, col=1
                    )
                    
                    # Hate
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['hate']['probabilities'].keys()),
                            y=list(prediction['hate']['probabilities'].values()),
                            name='Ngôn từ thù địch',
                            marker_color='red'
                        ),
                        row=1, col=2
                    )
                    
                    # Violence
                    fig_combined.add_trace(
                        go.Bar(
                            x=list(prediction['violence']['probabilities'].keys()),
                            y=list(prediction['violence']['probabilities'].values()),
                            name='Bạo lực',
                            marker_color='orange'
                        ),
                        row=1, col=3
                    )
                    
                    fig_combined.update_layout(
                        height=400,
                        showlegend=False,
                        title_text="Phân tích Multi-Task"
                    )
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
        
        if st.session_state.model_loaded:
            model_info = st.session_state.model_service.get_model_info()
            
            st.subheader("📊 Model Information")
            st.json(model_info)
            
            st.subheader("📈 Task Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎭 Emotion Classes**")
                for i, cls in enumerate(model_info['emotion_classes'], 1):
                    st.write(f"{i}. {cls}")
            
            with col2:
                st.markdown("**😡 Hate Classes**")
                for i, cls in enumerate(model_info['hate_classes'], 1):
                    st.write(f"{i}. {cls}")
            
            with col3:
                st.markdown("**⚔️ Violence Classes**")
                for i, cls in enumerate(model_info['violence_classes'], 1):
                    st.write(f"{i}. {cls}")
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

