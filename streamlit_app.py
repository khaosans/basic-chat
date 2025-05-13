import streamlit as st
import os
import time
import requests
import json
import base64
from PIL import Image
import io
from typing import List, Dict, Optional
import tempfile
from datetime import datetime
import urllib.request

# Set page configuration
st.set_page_config(
    page_title="Multimodal Evaluator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava")  # Default to LLaVA for multimodal

# Sample images for auto-testing
SAMPLE_IMAGES = {
    "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=500",
    "city": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?q=80&w=500",
    "food": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=500"
}

# Sample text prompts for auto-testing
SAMPLE_TEXT_PROMPTS = [
    {
        "prompt": "What is machine learning and how does it work?",
        "expected_elements": ["algorithm", "data", "training", "model", "prediction"]
    },
    {
        "prompt": "Explain the concept of climate change and its impacts.",
        "expected_elements": ["global warming", "greenhouse gases", "temperature", "sea level", "emissions"]
    },
    {
        "prompt": "What are the key features of Python programming language?",
        "expected_elements": ["interpreted", "dynamic typing", "object-oriented", "high-level", "readability"]
    }
]

# Sample image prompts for auto-testing
SAMPLE_IMAGE_PROMPTS = [
    {
        "image_key": "cat",
        "prompt": "Describe this image in detail.",
        "expected_elements": ["cat", "animal", "pet", "fur", "whiskers"]
    },
    {
        "image_key": "city",
        "prompt": "What can you tell me about this urban landscape?",
        "expected_elements": ["buildings", "skyline", "urban", "architecture", "city"]
    },
    {
        "image_key": "food",
        "prompt": "Identify the items in this food image.",
        "expected_elements": ["vegetables", "healthy", "colorful", "fresh", "salad"]
    }
]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []
if "auto_test_running" not in st.session_state:
    st.session_state.auto_test_running = False
if "auto_test_results" not in st.session_state:
    st.session_state.auto_test_results = []
if "cached_images" not in st.session_state:
    st.session_state.cached_images = {}
# Add new state variables for chatbot mode
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None  # None, "testing", or "chatbot"
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = []

class MultimodalEvaluator:
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name
        self.api_url = f"{OLLAMA_API_URL}/generate"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 encoding for API request"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def process_image_from_memory(self, image_bytes: bytes) -> str:
        """Process image directly from memory"""
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def query_with_image(self, prompt: str, image_data: bytes) -> Optional[str]:
        """Send a query with both text and image to the model"""
        try:
            # Convert image to base64
            base64_image = self.process_image_from_memory(image_data)
            
            # Prepare the payload for Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [base64_image]
            }
            
            # Make the API request
            with st.spinner("Processing image and generating response..."):
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "No response received")
                
        except Exception as e:
            st.error(f"Error querying model: {str(e)}")
            return None
    
    def query_text_only(self, prompt: str) -> Optional[str]:
        """Send a text-only query to the model"""
        try:
            # Prepare the payload for Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Make the API request
            with st.spinner("Generating response..."):
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "No response received")
                
        except Exception as e:
            st.error(f"Error querying model: {str(e)}")
            return None

def evaluate_response(query: str, response: str, expected_elements: List[str], image_data: bytes = None) -> Dict:
    """Evaluate the model's response based on expected elements"""
    score = 0
    max_score = len(expected_elements)
    found_elements = []
    
    for element in expected_elements:
        if element.lower() in response.lower():
            score += 1
            found_elements.append(element)
    
    # Add pass/fail status
    status = "PASS" if score == max_score else "PARTIAL" if score > 0 else "FAIL"
    
    result = {
        "query": query,
        "response": response,
        "expected_elements": expected_elements,
        "found_elements": found_elements,
        "score": score,
        "max_score": max_score,
        "percentage": round((score / max_score) * 100, 2) if max_score > 0 else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status
    }
    
    # Store image data if provided
    if image_data is not None:
        result["has_image"] = True
        result["image_data"] = base64.b64encode(image_data).decode("utf-8")
    else:
        result["has_image"] = False
    
    return result

def display_chat_message(role: str, content: str, image=None):
    """Display a chat message with optional image"""
    with st.chat_message(role):
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(content)

def download_image(url: str) -> bytes:
    """Download image from URL and return as bytes"""
    # Check if image is already cached
    if url in st.session_state.cached_images:
        return st.session_state.cached_images[url]
    
    # Download the image
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        
    # Cache the image
    st.session_state.cached_images[url] = image_data
    return image_data

def set_active_mode(mode: str):
    """Set the active mode and disable other interactions"""
    st.session_state.active_mode = mode

def run_auto_test(evaluator: MultimodalEvaluator, test_type: str = "all"):
    """Run automated tests with sample data"""
    set_active_mode("testing")
    st.session_state.auto_test_running = True
    st.session_state.auto_test_results = []
    
    with st.spinner(f"Running automated {test_type} tests..."):
        # Run text tests if requested
        if test_type in ["all", "text"]:
            for test_case in SAMPLE_TEXT_PROMPTS:
                prompt = test_case["prompt"]
                expected_elements = test_case["expected_elements"]
                
                # Display the user message
                display_chat_message("user", prompt)
                
                # Get response from the model
                response = evaluator.query_text_only(prompt)
                
                if response:
                    # Display the assistant's response
                    display_chat_message("assistant", response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": prompt,
                        "has_image": False
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "has_image": False
                    })
                    
                    # Evaluate the response
                    eval_result = evaluate_response(prompt, response, expected_elements)
                    st.session_state.evaluation_history.append(eval_result)
                    st.session_state.auto_test_results.append(eval_result)
                    
                    # Display evaluation result
                    st.success(f"Text Test: {eval_result['score']}/{eval_result['max_score']} ({eval_result['percentage']}%)")
                    
                    # Add a small delay to avoid overwhelming the API
                    time.sleep(1)
        
        # Run image tests if requested
        if test_type in ["all", "image"]:
            for test_case in SAMPLE_IMAGE_PROMPTS:
                image_key = test_case["image_key"]
                prompt = test_case["prompt"]
                expected_elements = test_case["expected_elements"]
                
                # Download the image
                image_url = SAMPLE_IMAGES[image_key]
                image_data = download_image(image_url)
                
                # Display the user message with image
                pil_image = Image.open(io.BytesIO(image_data))
                display_chat_message("user", prompt, pil_image)
                
                # Process the image and get response
                response = evaluator.query_with_image(prompt, image_data)
                
                if response:
                    # Display the assistant's response
                    display_chat_message("assistant", response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": prompt,
                        "has_image": True
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "has_image": False
                    })
                    
                    # Evaluate the response - pass the image data
                    eval_result = evaluate_response(prompt, response, expected_elements, image_data)
                    st.session_state.evaluation_history.append(eval_result)
                    st.session_state.auto_test_results.append(eval_result)
                    
                    # Display evaluation result
                    st.success(f"Image Test ({image_key}): {eval_result['score']}/{eval_result['max_score']} ({eval_result['percentage']}%)")
                    
                    # Add a small delay to avoid overwhelming the API
                    time.sleep(1)
    
    st.session_state.auto_test_running = False
    set_active_mode(None)
    
    # Calculate and display overall results
    if st.session_state.auto_test_results:
        avg_score = sum(result["percentage"] for result in st.session_state.auto_test_results) / len(st.session_state.auto_test_results)
        st.metric("Overall Auto-Test Score", f"{avg_score:.2f}%")

def main():
    st.title("🧪 Multimodal Evaluation Testing")
    st.subheader("Test your AI's ability to understand and respond to different types of media")
    
    # Initialize the evaluator
    evaluator = MultimodalEvaluator(OLLAMA_MODEL)
    
    # Sidebar for configuration and results
    with st.sidebar:
        # Add custom CSS to improve sidebar spacing and readability
        st.markdown("""
        <style>
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        .sidebar h1, .sidebar h2, .sidebar h3 {
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid rgba(250, 250, 250, 0.2);
        }
        .sidebar .stRadio > div {
            padding: 10px 0;
        }
        .sidebar .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Configuration section with better spacing
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        model_selection = st.selectbox(
            "Select Model",
            ["llava", "mistral"],
            index=0,
            help="Choose the model to test. LLaVA supports images, Mistral is text-only."
        )
        
        if model_selection != OLLAMA_MODEL:
            evaluator = MultimodalEvaluator(model_selection)
        
        # Display active mode status with better visibility
        if st.session_state.active_mode:
            st.markdown("---")
            st.warning(f"Active Mode: {st.session_state.active_mode.capitalize()}")
            if st.button("Cancel Current Operation"):
                set_active_mode(None)
                st.session_state.auto_test_running = False
                st.rerun()
            st.markdown("---")
        
        # Automated testing section with better organization
        st.header("🤖 Automated Testing")
        st.markdown("---")
        
        # Group related buttons together
        st.subheader("Test Controls")
        auto_test_col1, auto_test_col2 = st.columns(2)
        
        with auto_test_col1:
            if st.button("Run All Tests", disabled=st.session_state.active_mode is not None):
                run_auto_test(evaluator, "all")
        
        with auto_test_col2:
            if st.button("Clear Results"):
                st.session_state.evaluation_history = []
                st.session_state.auto_test_results = []
                st.rerun()
        
        # Test type selection with better spacing
        st.subheader("Test Type")
        test_type = st.radio("Select test category:", ["All", "Text Only", "Image Only"])
        
        if st.button("Run Selected Tests", disabled=st.session_state.active_mode is not None):
            if test_type == "All":
                run_auto_test(evaluator, "all")
            elif test_type == "Text Only":
                run_auto_test(evaluator, "text")
            elif test_type == "Image Only":
                run_auto_test(evaluator, "image")
        
        # Results section with better visibility
        st.markdown("---")
        st.header("📊 Evaluation Results")
        
        if st.session_state.evaluation_history:
            avg_score = sum(eval_item["percentage"] for eval_item in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
            st.metric("Average Score", f"{avg_score:.2f}%")
            
            # Add a download button for results
            if st.button("Download Results"):
                # Create a JSON string of results
                results_json = json.dumps(st.session_state.evaluation_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name="evaluation_results.json",
                    mime="application/json"
                )
        else:
            st.info("No evaluation results yet")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Auto Testing", "Image Understanding", "Text Analysis", "Evaluation History", "Chatbot"])
    
    # Define interaction_disabled variable based on active mode
    interaction_disabled = st.session_state.active_mode is not None
    
    with tab1:
        st.header("🤖 Automated Multimodal Testing")
        st.write("Run pre-configured tests to evaluate your model's multimodal capabilities.")
        
        # Display auto-test configuration
        st.subheader("Test Configuration")
        
        st.write("**Text Test Cases:**")
        for i, test in enumerate(SAMPLE_TEXT_PROMPTS):
            with st.expander(f"Text Test #{i+1}: {test['prompt'][:50]}..."):
                st.write(f"**Prompt:** {test['prompt']}")
                st.write(f"**Expected Elements:** {', '.join(test['expected_elements'])}")
        
        st.write("**Image Test Cases:**")
        for i, test in enumerate(SAMPLE_IMAGE_PROMPTS):
            with st.expander(f"Image Test #{i+1}: {test['prompt'][:50]}..."):
                st.write(f"**Image:** {test['image_key']}")
                st.image(SAMPLE_IMAGES[test['image_key']], width=200)
                st.write(f"**Prompt:** {test['prompt']}")
                st.write(f"**Expected Elements:** {', '.join(test['expected_elements'])}")
        
        # Auto-run tests on page load if requested
        auto_run = st.checkbox("Auto-run tests on page load", value=False)
        if auto_run and not st.session_state.auto_test_running and not st.session_state.auto_test_results:
            run_auto_test(evaluator, "all")
        
        # Display auto-test results
        if st.session_state.auto_test_results:
            st.subheader("Auto-Test Results")
            
            # Calculate overall statistics
            total_tests = len(st.session_state.auto_test_results)
            avg_score = sum(result["percentage"] for result in st.session_state.auto_test_results) / total_tests
            text_results = [r for r in st.session_state.auto_test_results if not any(img_key in r["query"] for img_key in SAMPLE_IMAGES.keys())]
            image_results = [r for r in st.session_state.auto_test_results if any(img_key in r["query"] for img_key in SAMPLE_IMAGES.keys())]
            
            text_avg = sum(r["percentage"] for r in text_results) / len(text_results) if text_results else 0
            image_avg = sum(r["percentage"] for r in image_results) / len(image_results) if image_results else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{avg_score:.2f}%")
            with col2:
                st.metric("Text Score", f"{text_avg:.2f}%")
            with col3:
                st.metric("Image Score", f"{image_avg:.2f}%")
            
            # Display detailed results
            for i, result in enumerate(st.session_state.auto_test_results):
                with st.expander(f"Test #{i+1}: {result['query'][:50]}... ({result['percentage']}%)"):
                    st.write(f"**Query:** {result['query']}")
                    st.write(f"**Response:** {result['response']}")
                    st.write(f"**Expected Elements:** {', '.join(result['expected_elements'])}")
                    st.write(f"**Found Elements:** {', '.join(result['found_elements'])}")
                    st.write(f"**Score:** {result['score']}/{result['max_score']} ({result['percentage']}%)")
    
    with tab2:
        st.header("🖼️ Image Understanding Test")
        st.write("Upload an image and ask the model to describe or analyze it.")
        
        # Use the interaction_disabled variable to control UI elements
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], disabled=interaction_disabled)
        image_prompt = st.text_input("Ask a question about the image", placeholder="What can you see in this image?", disabled=interaction_disabled)
        
        expected_elements = st.text_area(
            "Expected elements in response (one per line)", 
            placeholder="Enter elements you expect to see in the response, one per line",
            disabled=interaction_disabled
        ).strip().split("\n") if st.text_area("Expected elements in response (one per line)", placeholder="Enter elements you expect to see in the response, one per line", disabled=interaction_disabled) else []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Image Understanding", disabled=not (uploaded_image and image_prompt) or interaction_disabled):
                # Set active mode to testing
                set_active_mode("testing")
                
                if uploaded_image:
                    # Display the user message with image
                    display_chat_message("user", image_prompt, Image.open(uploaded_image))
                    
                    # Process the image and get response
                    image_bytes = uploaded_image.getvalue()
                    response = evaluator.query_with_image(image_prompt, image_bytes)
                    
                    if response:
                        # Display the assistant's response
                        display_chat_message("assistant", response)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": image_prompt,
                            "has_image": True
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "has_image": False
                        })
                        
                        # Evaluate the response if expected elements are provided
                        if expected_elements and expected_elements[0]:  # Check if there's at least one non-empty element
                            eval_result = evaluate_response(image_prompt, response, expected_elements, image_bytes)
                            st.session_state.evaluation_history.append(eval_result)
                            
                            # Display evaluation result with status badge
                            status = eval_result.get("status", "UNKNOWN")
                            status_color = "#28a745" if status == "PASS" else "#ffc107" if status == "PARTIAL" else "#dc3545"
                            status_text_color = "white" if status != "PARTIAL" else "black"
                            
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0;">Test Result</h4>
                                    <span style="background-color: {status_color}; color: {status_text_color}; padding: 5px 10px; border-radius: 3px; font-weight: bold;">{status}</span>
                                </div>
                                <p>Score: {eval_result['score']}/{eval_result['max_score']} ({eval_result['percentage']}%)</p>
                                <p><strong>Found Elements:</strong> {', '.join(eval_result['found_elements']) if eval_result['found_elements'] else 'None'}</p>
                                <p><strong>Missing Elements:</strong> {', '.join([elem for elem in expected_elements if elem not in eval_result['found_elements']]) if [elem for elem in expected_elements if elem not in eval_result['found_elements']] else 'None'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Reset active mode
                    set_active_mode(None)
    
    with tab3:
        st.header("📝 Text Analysis Test")
        st.write("Test the model's ability to understand and respond to text prompts.")
        
        # Use the interaction_disabled variable to control UI elements
        text_prompt = st.text_area("Enter your text prompt", placeholder="Ask a question or provide a task...", disabled=interaction_disabled)
        
        expected_text_elements = st.text_area(
            "Expected elements in text response (one per line)", 
            placeholder="Enter elements you expect to see in the response, one per line"
        ).strip().split("\n") if st.text_area("Expected elements in text response (one per line)", placeholder="Enter elements you expect to see in the response, one per line", disabled=interaction_disabled) else []
        
        if st.button("Test Text Understanding", disabled=not text_prompt or interaction_disabled):
            # Set active mode to testing
            set_active_mode("testing")
            
            # Display the user message
            display_chat_message("user", text_prompt)
            
            # Get response from the model
            response = evaluator.query_text_only(text_prompt)
            
            if response:
                # Display the assistant's response
                display_chat_message("assistant", response)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": text_prompt,
                    "has_image": False
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "has_image": False
                })
                
                # Evaluate the response if expected elements are provided
                if expected_text_elements and expected_text_elements[0]:  # Check if there's at least one non-empty element
                    eval_result = evaluate_response(text_prompt, response, expected_text_elements)
                    st.session_state.evaluation_history.append(eval_result)
                    
                    # Display evaluation result with improved styling
                    status_color = "#28a745" if eval_result["status"] == "PASS" else "#ffc107" if eval_result["status"] == "PARTIAL" else "#dc3545"
                    status_text_color = "white" if eval_result["status"] != "PARTIAL" else "black"
                    
                    st.markdown(f"""
                    <div class="test-card">
                        <div class="test-card-header">
                            <h4>Test Result</h4>
                            <div class="status-badge" style="background-color: {status_color}; color: {status_text_color};">{eval_result["status"]}</div>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {eval_result['percentage']}%; background-color: {status_color};"></div>
                        </div>
                        <p>Score: {eval_result['score']}/{eval_result['max_score']} ({eval_result['percentage']}%)</p>
                        <div class="test-card-content">
                            <p><strong>Found Elements:</strong> {', '.join(eval_result['found_elements']) if eval_result['found_elements'] else 'None'}</p>
                            <p><strong>Missing Elements:</strong> {', '.join([elem for elem in expected_text_elements if elem not in eval_result['found_elements']]) if [elem for elem in expected_text_elements if elem not in eval_result['found_elements']] else 'None'}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Reset active mode
            set_active_mode(None)
    
    with tab4:
        st.header("📊 Evaluation History")
        st.write("View the history of all evaluation tests run in this session.")
        
        if not st.session_state.evaluation_history:
            st.info("No evaluation results yet. Run some tests to see results here.")
        else:
            # Add filter options
            filter_options = ["All", "PASS", "PARTIAL", "FAIL", "With Images", "Text Only"]
            selected_filter = st.selectbox("Filter results:", filter_options)
            
            # Filter results based on selection
            filtered_results = st.session_state.evaluation_history
            if selected_filter == "PASS":
                filtered_results = [r for r in st.session_state.evaluation_history if r.get("status") == "PASS"]
            elif selected_filter == "PARTIAL":
                filtered_results = [r for r in st.session_state.evaluation_history if r.get("status") == "PARTIAL"]
            elif selected_filter == "FAIL":
                filtered_results = [r for r in st.session_state.evaluation_history if r.get("status") == "FAIL"]
            elif selected_filter == "With Images":
                filtered_results = [r for r in st.session_state.evaluation_history if r.get("has_image", False)]
            elif selected_filter == "Text Only":
                filtered_results = [r for r in st.session_state.evaluation_history if not r.get("has_image", False)]
            
            # Display evaluation history
            for i, eval_item in enumerate(filtered_results):
                # Determine status and colors
                status = eval_item.get("status", "UNKNOWN")
                status_color = "#28a745" if status == "PASS" else "#ffc107" if status == "PARTIAL" else "#dc3545"
                status_text_color = "white" if status != "PARTIAL" else "black"
                
                with st.expander(f"Evaluation #{i+1}: {eval_item['query'][:50]}... ({eval_item['percentage']}%)"):
                    # Display image if available
                    if eval_item.get("has_image", False) and "image_data" in eval_item:
                        image_bytes = base64.b64decode(eval_item["image_data"])
                        st.image(Image.open(io.BytesIO(image_bytes)), caption="Test Image", width=300)
                    
                    # Display status badge
                    st.markdown(f"""
                    <span style="background-color: {status_color}; color: {status_text_color}; padding: 5px 10px; border-radius: 3px; font-weight: bold;">{status}</span>
                    """, unsafe_allow_html=True)
                    
                    # Display score with progress bar
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <p>Score: {eval_item['score']}/{eval_item['max_score']} ({eval_item['percentage']}%)</p>
                        <div style="background-color: #e9ecef; border-radius: 4px; height: 8px; width: 100%;">
                            <div style="background-color: {status_color}; width: {eval_item['percentage']}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display other details
                    st.write(f"**Query:** {eval_item['query']}")
                    st.write(f"**Response:** {eval_item['response']}")
                    st.write(f"**Expected Elements:** {', '.join(eval_item['expected_elements'])}")
                    st.write(f"**Found Elements:** {', '.join(eval_item['found_elements'])}")
                    st.write(f"**Missing Elements:** {', '.join([elem for elem in eval_item['expected_elements'] if elem not in eval_item['found_elements']])}")
                    st.write(f"**Timestamp:** {eval_item['timestamp']}")
    
    # New Chatbot Tab
    with tab5:
        st.header("💬 AI Chatbot")
        st.write("Have a conversation with the AI model.")
        
        # Display chat history
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                if "image" in message and message["image"] is not None:
                    st.image(message["image"], caption="Uploaded Image")
                st.write(message["content"])
        
        # Chat input area
        chat_col1, chat_col2 = st.columns([3, 1])
        
        with chat_col1:
            user_input = st.chat_input("Type your message here...", disabled=interaction_disabled)
        
        with chat_col2:
            chat_image = st.file_uploader("Upload image for chat", type=["jpg", "jpeg", "png"], disabled=interaction_disabled, key="chat_image")
        
        # Process user input
        if user_input:
            # Set active mode to chatbot
            if st.session_state.active_mode is None:
                set_active_mode("chatbot")
            
            # Add user message to chat history
            user_message = {"role": "user", "content": user_input}
            if chat_image:
                img = Image.open(chat_image)
                user_message["image"] = img
            st.session_state.chatbot_messages.append(user_message)
            
            # Display user message
            with st.chat_message("user"):
                if chat_image:
                    st.image(Image.open(chat_image), caption="Uploaded Image")
                st.write(user_input)
            
            # Generate AI response
            with st.spinner("AI is thinking..."):
                if chat_image:
                    # Process image chat
                    image_bytes = chat_image.getvalue()
                    response = evaluator.query_with_image(user_input, image_bytes)
                else:
                    # Process text-only chat
                    response = evaluator.query_text_only(user_input)
            
            if response:
                # Add AI response to chat history
                st.session_state.chatbot_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.write(response)
            
            # Reset active mode
            set_active_mode(None)
            
            # Force a rerun to update the UI
            st.rerun()
        
        # Add buttons to clear chat or start a new conversation
        chat_buttons_col1, chat_buttons_col2 = st.columns(2)
        
        with chat_buttons_col1:
            if st.button("Clear Chat History", disabled=interaction_disabled):
                st.session_state.chatbot_messages = []
                st.rerun()
        
        with chat_buttons_col2:
            if st.button("Start New Conversation", disabled=interaction_disabled):
                st.session_state.chatbot_messages = []
                st.rerun()

if __name__ == "__main__":
    main()