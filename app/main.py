import streamlit as st
import constants
import pandas as pd
from terrain import Terrain
import matplotlib.colors
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip  # to produce mp4 video

st.set_page_config(page_title="Terrain Generation",
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
    
def main():
    remove_top_margin()
    width_perc = constants.WIDTH_RESIZE_PERC
    side_perc = (100 - width_perc)/2    
    _, center_container, _ = st.columns([side_perc, 
                                        width_perc, 
                                        side_perc])    
    center_container.title("Terrain Generation")    
    side_con = st.sidebar.container()
    
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
                                options = ["Evolution", "Morphing"],
                                key = "method_elem"
                                )
    
    # options: 'Storm', 'Terrain' or 'Vincent'
    palette = side_con.selectbox(label = "Palette", 
                                options = ["Storm", "Terrain",
                                           "Vincent"],
                                key = "palette_elem"
                                )    
    
    # options: 'Blending' or 'Mixture'
    mode = side_con.selectbox(label = "Mode", 
                              options = ["Blending", "Mixture"],
                              key = "mode_elem"
                              )
    
    if mode == "Mixture":
        distribution = side_con.selectbox(label = "Distribution", 
                                          options = ["Gaussian", "Uniform"],
                                          key = "distribution_elem"
                                          )
    else:
        distribution = side_con.selectbox(label = "Distribution", 
                                          options = ["Gaussian"],
                                          key = "distribution_elem"
                                          )
    
    bdry = side_con.selectbox(label = "Averaging", 
                              options = ["periodic", "fixed"],
                              key = "averaging_elem"
                              )
    
    ds = side_con.slider(label = "Delta", 
                         min_value=0.0,
                         max_value=1.0, 
                         step = 0.01,
                         value = 0.8,
                         key = "delta_elem"
                         )
        
    jump = side_con.slider(label = "Jump", 
                         min_value=0.0,
                         max_value=1.0, 
                         step = 0.01,
                         value = 0.5,
                         key = "jump_elem"
                         )

    n = side_con.selectbox(label = "Edge Size", 
                           options = [32, 64, 128, 256, 512],
                           index = 3,
                           key = "edge_size_elem"
                         ) + 1
    
    Nframes = side_con.number_input(label = "Num Frames", 
                                    min_value = 1, 
                                    max_value = 10000,
                                    step = 1,
                                    value = 10)
    
    start_seed = side_con.number_input(
        "Start Seed",
        min_value = 0,
        value = 134,
        step = 1,
        key = "start_seed_elem"
        )
    
    if method == "Morphing":
        end_seed = side_con.number_input(
            "End Seed",
            min_value = 0,
            value = 143,            
            step = 1,
            key = "end_seed_elem"
            )        
    
    run_button = st.sidebar.button("Submit", type="primary")
    
    if run_button:
        params = {
            "method": method, "palette": palette, "mode": mode,
            "distribution": distribution,  "weight": weight,
            "dpi": dpi, "n": n, "ds": ds, "bdry": bdry, 
            "Nframes": Nframes, "image_dir": image_dir, 
            "start_seed": start_seed, "end_seed": end_seed if method == "Morphing" else None, "jump": jump,
            "col_morphing": col_morphing
        }
        
        try:
            terrain = Terrain(params)
            terrain.run()
            # output video

            fps = constants.VIDEO_FRAMES_PER_SEC
            with center_container, st.spinner("Generating Video File..."):
                clip = \
                moviepy.video.io.ImageSequenceClip.ImageSequenceClip(terrain.flist, fps=fps)
                video_file = f"{image_dir}/terrain.mp4"
                clip.write_videofile(video_file)
            center_container.video(video_file)
        except Exception as e:
            st.error(f"Error!! {str(e)}")
    
    