import streamlit as st
from typing import List
import constants
import pandas as pd
import logging
from terrain import Terrain
import matplotlib.colors
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip  # to produce mp4 video
from moviepy.editor import ImageSequenceClip
import numpy as np
from PIL import Image
import time
import io


# logging.basicConfig(level=logging.DEBUG)
title = constants.TITLE
container_dict = dict(video_container = None)
width_perc = constants.WIDTH_RESIZE_PERC
side_perc = (100 - width_perc)/2  
st.set_page_config(page_title=title,
                   page_icon="",
                   layout="wide")
  
def remove_top_margin():
    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=0, padding_bottom=5
        ),
        unsafe_allow_html=True
    )
    
def generate_video(video_container, image_dir: str, 
                   image_list: List, fps: int):
    with video_container, st.spinner("Generating Video File..."):
        # Use this logic if image_list is list of image file names
        # clip = \
        # moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list, fps=fps)
        clip = ImageSequenceClip([np.array(img) for img in image_list], fps=fps)        
        video_file = f"{image_dir}/terrain.mp4"
        clip.write_videofile(video_file)
    video_container.video(video_file)
    
def generate_images(image_center, params):
    image_list = []
    terrain = Terrain(params)
    progress_message = f"Generating Images for {terrain.method}"
    progress_bar = \
        image_center.progress(0, f"{progress_message}..Completed 0%")
    image_count = 0
    image_placeholder = image_center.empty()
    for result in terrain.run():
        # st.write(result)
        progress_perc = result["progress_perc"]
        img = result["image"]
        image_placeholder.image(img)
        progress_bar.progress(progress_perc, 
                                f"{progress_message}..Completed {progress_perc}%")        
        image_list.append(img)
        time.sleep(0.2)

    progress_bar.empty()
    image_placeholder.empty()
    return image_list
        
    
def initialize_sessions():
    #Initialise the key in session state
    if "clicked" not in st.session_state:
        st.session_state.clicked = {1:False}
                
def click_submit(button):
    st.session_state.clicked[button] = True

def reset_actions():
    st.session_state.method_elem = constants.METHOD_OPTIONS[0]
    st.session_state.palette_elem = constants.PALETTE_OPTIONS[0]
    st.session_state.mode_elem = constants.MODE_OPTIONS[0]
    st.session_state.distribution_elem = \
        constants.MIXTURE_DISTRIBUTION_OPTIONS[0]
    st.session_state.averaging_elem = list(constants.AVERAGING_OPTIONS.keys())[0]
    st.session_state.delta_elem = constants.DELTA_DEFAULT
    st.session_state.jump_elem = constants.JUMP_DEFAULT
    st.session_state.edge_size_elem = constants.EDGE_SIZE_OPTIONS[3]
    st.session_state.nframes_elem = constants.NUM_FRAMES_DEFAULT
    st.session_state.start_seed_elem = constants.START_SEED_DEFAULT
    st.session_state.end_seed_elem = constants.END_SEED_DEFAULT
    
def number_initialize():
    if "edge_size_elem" not in st.session_state:
        st.session_state.edge_size_elem = constants.EDGE_SIZE_OPTIONS[3]
    
    if "delta_elem" not in st.session_state:
        st.session_state.delta_elem = constants.DELTA_DEFAULT    

    if "jump_elem" not in st.session_state:
        st.session_state.jump_elem = constants.JUMP_DEFAULT    

    if "nframes_elem" not in st.session_state:
        st.session_state.nframes_elem = constants.NUM_FRAMES_DEFAULT    

    if "start_seed_elem" not in st.session_state:
        st.session_state.start_seed_elem = constants.START_SEED_DEFAULT    

    if "end_seed_elem" not in st.session_state:
        st.session_state.end_seed_elem = constants.END_SEED_DEFAULT

    if "dummy" not in st.session_state:
        st.session_state.dummy = 0
    
def main():
    remove_top_margin()       
    initialize_sessions()
    number_initialize()
    _, title_container, _ = st.columns([side_perc, 
                                        width_perc, 
                                        side_perc])     
    title_container.title(title)    
    side_con = st.sidebar.container()
    side_con.button("Reset", type="primary",
                      on_click=reset_actions)
    
    # _, image_container, _ = body.columns([side_perc, 
    #                                     width_perc, 
    #                                     side_perc])    
    
    # used in Gaussian mixture: low weight keeps pixel color little changed
    weight = constants.GAUSSIAN_MIXTURE_WEIGHT
    
    # dots per inch (image resolution)
    dpi = constants.DOTS_PER_INCH
    
    # image dir
    image_dir = constants.IMAGE_DIR
    
    # Available in morphing method only. False for now
    col_morphing = constants.COLUMN_MORPHING
    
    # method: 'Morphing' or 'Evolution'
    method = side_con.selectbox(label = "Method", 
                                options = constants.METHOD_OPTIONS,
                                key = "method_elem"
                                )
    
    # options: 'Storm', 'Terrain' or 'Vincent'
    palette = side_con.selectbox(label = "Palette", 
                                options = constants.PALETTE_OPTIONS,
                                key = "palette_elem"
                                )    
    
    # options: 'Blending' or 'Mixture'
    mode = side_con.selectbox(label = "Mode", 
                              options = constants.MODE_OPTIONS,
                              key = "mode_elem"
                              )
    
    if mode == "Mixture":
        distribution = side_con.selectbox(label = "Distribution", 
                                          options = constants.MIXTURE_DISTRIBUTION_OPTIONS,
                                          key = "distribution_elem"
                                          )
    else:
        distribution = side_con.selectbox(label = "Distribution", 
                                          options = constants.BLENDING_DISTRIBUTION_OPTIONS,
                                          key = "distribution_elem"
                                          )
    
    bdry_select = \
        side_con.selectbox(label = "Averaging", 
                           options = list(constants.AVERAGING_OPTIONS.keys()),
                           key = "averaging_elem"
                           )
    bdry = constants.AVERAGING_OPTIONS[bdry_select]
    
    ds = side_con.slider(label = "Delta", 
                         min_value=0.0,
                         max_value=1.0, 
                         step = 0.01,
                         key = "delta_elem"
                         )
        
    jump = side_con.slider(label = "Jump", 
                         min_value=0.0,
                         max_value=1.0, 
                         step = 0.01,
                         key = "jump_elem"
                         )

    n = side_con.selectbox(label = "Edge Size", 
                           options = constants.EDGE_SIZE_OPTIONS,
                           key = "edge_size_elem"
                         ) + 1
    
    Nframes = side_con.number_input(label = "Num Frames", 
                                    min_value = 1, 
                                    max_value = 30,
                                    step = 1,
                                    key = "nframes_elem")
    
    start_seed = side_con.number_input(
        "Start Seed",
        min_value = 1,
        step = 1,
        key = "start_seed_elem"
        )
    
    if method == "Morphing":
        if "end_seed_elem" not in st.session_state:
            st.session_state.end_seed_elem = constants.END_SEED_DEFAULT 
            
        end_seed = side_con.number_input(
            "End Seed",
            min_value = 1,           
            step = 1,
            key = "end_seed_elem"
            )        
    
    # _, video_container, _ = st.columns([side_perc, 
    #                             width_perc, 
    #                             side_perc])

    run_button = st.sidebar.button("Submit", 
                                   on_click = click_submit,
                                   args = [1],
                                   type="primary")
    if st.session_state.clicked[1]:
        st.session_state.clicked[1] = False
            
        params = {
            "method": method, "palette": palette, "mode": mode,
            "distribution": distribution,  "weight": weight,
            "dpi": dpi, "n": n, "ds": ds, "bdry": bdry, 
            "Nframes": Nframes, "image_dir": image_dir, 
            "start_seed": start_seed, "end_seed": end_seed if method == "Morphing" else None, "jump": jump,
            "col_morphing": col_morphing
        }
        
        try:
            #Generate Images
            _, image_center, _ = st.columns([side_perc, 
                                            width_perc, 
                                            side_perc])            
            image_list = generate_images(image_center,
                                         params)            
            # output video

            fps = constants.VIDEO_FRAMES_PER_SEC
            # _, video_container, _ = \
            #     st.columns([side_perc, 
            #                 width_perc, 
            #                 side_perc])
                
            generate_video(image_center,
                            image_dir,
                            image_list, fps) 
        except Exception as e:
            st.error(f"Error!! {str(e)}")