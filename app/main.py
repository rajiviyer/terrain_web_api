import streamlit as st
import pandas as pd
# from logging import getLogger
# from terrain import evolution, morphing, set_palette
from terrain import Terrain
#log = getLogger("test")
#log.error("this is a test")
import matplotlib.colors
import matplotlib.pyplot as plt
# import moviepy.video.io.ImageSequenceClip  # to produce mp4 video

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
    st.title("Terrain Generation")    
    side_con = st.sidebar.container()
    
    # used in Gaussian mixture: low weight keeps pixel color little changed
    weight = 0.2
    
    # dots per inch (image resolution)
    dpi = 300
    
    # image dir
    image_dir = "./data"
    
    # Available in morphing method only. False for now
    col_morphing = False
    
    # method: 'Morphing' or 'Evolution'
    method = side_con.selectbox(label = "Method", 
                                options = ["Evolution", "Morphing"],
                                key = "method_elem"
                                )
    
    # options: 'Storm', 'Terrain' or 'Vincent'
    palette = side_con.selectbox(label = "Palette", 
                                options = ["Storm", "Terrain", "Vincent"],
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
        # color_table = set_palette(palette)
        # cm = matplotlib.colors.LinearSegmentedColormap.from_list('geo-smooth',color_table)
        # print(f"Nframes: {Nframes}")
        # if method == 'Evolution':
        #     evolution(start_seed)
        # elif method == 'Morphing':
        #     morphing(start_seed,end_seed)        
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
            st.write(f"Run {method} Successful")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    