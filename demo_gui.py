import os
import sys
import cv2
import glob
import copy
import torch
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path

# Import từ demo/vis.py
sys.path.append('./demo')
import vis
from vis import get_pose2D, get_pose3D, img2video

def main():
    st.set_page_config(
        page_title="GraphMLP 3D Pose Estimation Demo",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ GraphMLP 3D Pose Estimation Demo")
    st.markdown("Upload a video to generate 2D and 3D pose estimation")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    fix_z = st.sidebar.checkbox("Fix Z-axis", value=False, help="Fix the Z-axis for better visualization")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.sidebar.success(f"CUDA Available: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.warning("Running on CPU (slower processing)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for pose estimation"
    )
    
    if uploaded_file is not None:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("Frames", frame_count)
        with col4:
            st.metric("Resolution", f"{width}x{height}")
        
        # Show original video
        st.subheader("📹 Original Video")
        st.video(uploaded_file)
        
        # Process button
        if st.button("🚀 Start Pose Estimation", type="primary"):
            process_video(video_path, uploaded_file.name, fix_z, temp_dir)
        
        # Cleanup
        if st.button("🗑️ Clear"):
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            st.rerun()

# def process_video(video_path, video_name, fix_z, temp_dir):
#     """Process video for pose estimation"""
    
#     # Setup output directory
#     video_basename = os.path.splitext(video_name)[0]
#     output_dir = f'./demo/output/{video_basename}/'
    
#     # Progress tracking
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Step 1: Generate 2D pose
#         status_text.text("🔍 Generating 2D pose estimation...")
#         progress_bar.progress(20)
        
#         with st.spinner("Processing 2D pose detection..."):
#             get_pose2D(video_path, output_dir)
        
#         st.success("✅ 2D pose generation completed!")
#         progress_bar.progress(40)
        
#         # Step 2: Generate 3D pose
#         status_text.text("🎯 Generating 3D pose estimation...")
#         progress_bar.progress(60)
        
#         with st.spinner("Processing 3D pose estimation..."):
#             get_pose3D(video_path, output_dir, fix_z)
        
#         st.success("✅ 3D pose generation completed!")
#         progress_bar.progress(80)
        
#         # Step 3: Create output video
#         status_text.text("🎬 Creating output video...")
#         progress_bar.progress(90)
        
#         with st.spinner("Generating final video..."):
#             img2video(video_path, output_dir)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Processing completed!")
        
#         # Display results
#         display_results(output_dir, video_basename)
        
#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.exception(e)
def process_video(video_path, video_name, fix_z, temp_dir):
    """Process video for pose estimation"""
    
    # Setup output directory
    video_basename = os.path.splitext(video_name)[0]
    output_dir = f'./demo/output/{video_basename}/'
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Generate 2D pose
        status_text.text("🔍 Generating 2D pose estimation...")
        progress_bar.progress(20)
        
        with st.spinner("Processing 2D pose detection..."):
            get_pose2D(video_path, output_dir)
        
        st.success("✅ 2D pose generation completed!")
        progress_bar.progress(40)
        
        # Step 2: Generate 3D pose
        status_text.text("🎯 Generating 3D pose estimation...")
        progress_bar.progress(60)
        
        with st.spinner("Processing 3D pose estimation..."):
            get_pose3D(video_path, output_dir, fix_z)
        
        st.success("✅ 3D pose generation completed!")
        progress_bar.progress(80)
        
        # Step 3: Create output video
        status_text.text("🎬 Creating output video...")
        progress_bar.progress(90)
        
        with st.spinner("Generating final video..."):
            # Đặt biến global video_name trong module vis
            vis.video_name = video_basename
            vis.img2video(video_path, output_dir)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing completed!")
        
        # Display results
        display_results(output_dir, video_basename)
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)

def display_results(output_dir, video_name):
    """Display processing results"""
    
    st.subheader("📊 Results")
    
    # Check if output video exists
    output_video_path = f"{output_dir}{video_name}.mp4"
    if os.path.exists(output_video_path):
        st.subheader("🎥 Result Video")
        with open(output_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
        
        # Download button
        st.download_button(
            label="📥 Download Result Video",
            data=video_bytes,
            file_name=f"{video_name}_pose_estimation.mp4",
            mime="video/mp4"
        )
    
    # Display sample frames
    display_sample_frames(output_dir)
    
    # Display 3D keypoints data
    display_3d_data(output_dir)

# def display_sample_frames(output_dir):
#     """Display sample frames from processing"""
    
#     # 2D pose frames
#     pose2d_dir = output_dir + 'pose2D/'
#     pose3d_dir = output_dir + 'pose3D/'
#     pose_dir = output_dir + 'pose/'
    
#     if os.path.exists(pose_dir):
#         st.subheader("🖼️ Sample Frames")
        
#         # Get sample frames
#         frame_files = sorted(glob.glob(os.path.join(pose_dir, '*.png')))
#         if frame_files:
#             # Show first, middle, and last frames
#             sample_indices = [0, len(frame_files)//2, -1]
            
#             cols = st.columns(3)
#             labels = ["First Frame", "Middle Frame", "Last Frame"]
            
#             for i, (col, idx, label) in enumerate(zip(cols, sample_indices, labels)):
#                 with col:
#                     st.text(label)
#                     frame_path = frame_files[idx]
#                     image = Image.open(frame_path)
#                     st.image(image, use_column_width=True)
    
#     # Show individual 2D and 3D poses
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if os.path.exists(pose2d_dir):
#             st.subheader("🎯 2D Pose Detection")
#             pose2d_files = sorted(glob.glob(os.path.join(pose2d_dir, '*.png')))
#             if pose2d_files:
#                 frame_idx = st.slider("Select 2D Frame", 0, len(pose2d_files)-1, 0)
#                 image = Image.open(pose2d_files[frame_idx])
#                 st.image(image, use_column_width=True)
    
#     with col2:
#         if os.path.exists(pose3d_dir):
#             st.subheader("🏃‍♂️ 3D Pose Estimation")
#             pose3d_files = sorted(glob.glob(os.path.join(pose3d_dir, '*.png')))
#             if pose3d_files:
#                 frame_idx = st.slider("Select 3D Frame", 0, len(pose3d_files)-1, 0)
#                 image = Image.open(pose3d_files[frame_idx])
#                 st.image(image, use_column_width=True)
def display_sample_frames(output_dir):
    """Display sample frames from processing"""
    
    # 2D pose frames
    pose2d_dir = output_dir + 'pose2D/'
    pose3d_dir = output_dir + 'pose3D/'
    pose_dir = output_dir + 'pose/'
    
    if os.path.exists(pose_dir):
        st.subheader("🖼️ Sample Frames")
        
        # Get sample frames
        frame_files = sorted(glob.glob(os.path.join(pose_dir, '*.png')))
        if frame_files:
            # Show first, middle, and last frames
            sample_indices = [0, len(frame_files)//2, -1]
            
            cols = st.columns(3)
            labels = ["First Frame", "Middle Frame", "Last Frame"]
            
            for i, (col, idx, label) in enumerate(zip(cols, sample_indices, labels)):
                with col:
                    st.text(label)
                    frame_path = frame_files[idx]
                    image = Image.open(frame_path)
                    st.image(image, width='stretch')
    
    # Show individual 2D and 3D poses
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(pose2d_dir):
            st.subheader("🎯 2D Pose Detection")
            pose2d_files = sorted(glob.glob(os.path.join(pose2d_dir, '*.png')))
            if pose2d_files:
                frame_idx = st.slider("Select 2D Frame", 0, len(pose2d_files)-1, 0)
                image = Image.open(pose2d_files[frame_idx])
                st.image(image, width='stretch')
    
    with col2:
        if os.path.exists(pose3d_dir):
            st.subheader("🏃‍♂️ 3D Pose Estimation")
            pose3d_files = sorted(glob.glob(os.path.join(pose3d_dir, '*.png')))
            if pose3d_files:
                frame_idx = st.slider("Select 3D Frame", 0, len(pose3d_files)-1, 0)
                image = Image.open(pose3d_files[frame_idx])
                st.image(image, width='stretch')

def display_3d_data(output_dir):
    """Display 3D keypoints data"""
    
    keypoints_3d_path = output_dir + 'output_3D/output_keypoints_3d.npz'
    if os.path.exists(keypoints_3d_path):
        st.subheader("📈 3D Keypoints Data")
        
        try:
            data = np.load(keypoints_3d_path, allow_pickle=True)
            keypoints_3d = data['reconstruction']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", keypoints_3d.shape[0])
            with col2:
                st.metric("Joints", keypoints_3d.shape[1])
            with col3:
                st.metric("Coordinates", keypoints_3d.shape[2])
            
            # Show data shape and sample
            st.text(f"Data shape: {keypoints_3d.shape}")
            
            # Download 3D data
            with open(keypoints_3d_path, 'rb') as f:
                st.download_button(
                    label="📥 Download 3D Keypoints Data",
                    data=f.read(),
                    file_name="keypoints_3d.npz",
                    mime="application/octet-stream"
                )
                
        except Exception as e:
            st.error(f"Error loading 3D keypoints: {str(e)}")

def add_footer():
    """Add footer with information"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>GraphMLP 3D Pose Estimation Demo</p>
        <p>Powered by Streamlit • PyTorch • OpenCV</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()