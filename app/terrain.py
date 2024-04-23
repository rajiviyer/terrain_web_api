# terrain8.py | www.MLTechniques.com | vincentg@MLTechniques.com | 2022

import random
import numpy as np
from typing import Dict
import streamlit as st
import matplotlib.colors
import matplotlib.pyplot as plt
import logging
import os
import constants

class Terrain():
    def __init__(self, params:Dict):
        self.method = params["method"]
        self.palette = params["palette"]
        self.mode = params["mode"]
        self.jump = params["jump"]
        self.distribution = params["distribution"]
        self.weight = params["weight"]
        self.n = params["n"]
        self.ds = params["ds"]
        self.bdry = params["bdry"]
        self.Nframes = params["Nframes"]
        self.dpi = params["dpi"]
        self.image_dir = params["image_dir"]
        self.image_file_prefix = "terrainM"
        self.image_file_suffix = ".png"
        self.start_seed = params["start_seed"]
        self.end_seed = params["end_seed"]
        self.col_morphing = params["col_morphing"]
        self.flist = []
        self.rnd_table  = {}

    def fixed(self, d, i, j, v, offsets ):
        # For fixed bdries, all cells are valid. Define n so as to allow the
        # usual lower bound inclusive, upper bound exclusive indexing.
        n = d.shape[0]
        
        res, k = 0, 0
        for p, q in offsets:
            pp, qq = i + p*v, j + q*v
            if 0 <= pp < n and 0 <= qq < n:
                res += d[pp, qq]
                k += 1.0
        return res/k

    def periodic(self, d, i, j, v, offsets ):
        # For periodic bdries, the last row/col mirrors the first row/col.
        # Hence the effective square size is (n-1)x(n-1). Redefine n accordingly!
        n = d.shape[0] - 1
        
        res = 0
        for p, q in offsets:
            res += d[(i + p*v)%n, (j + q*v)%n]
        return res/4.0

    def update_random_table(self, s, frame):

        global counter

        if self.distribution == 'Uniform' and self.mode == 'Blending':
            print("Error: Blending allowed only with Gaussian distribution.")
            exit()

        if frame < 1: 
            if self.distribution == 'Gaussian':
                self.rnd_table[counter]=random.gauss(0,s)
            elif self.distribution == 'Uniform':
                self.rnd_table[counter]=random.uniform(-s,s) 
        else:
            if random.uniform(0,1) > 1 - self.jump:
                if self.mode == 'Blending':
                    self.rnd_table[counter] += self.weight*random.gauss(0,s) # update random number table  # 0.5 * ..
                    self.rnd_table[counter] /= np.sqrt(1 + self.weight*self.weight)
                elif self.mode == 'Mixture':
                    if self.distribution == 'Gaussian':
                        self.rnd_table[counter] = random.gauss(0,s)
                    elif self.distribution == 'Uniform':
                        self.rnd_table[counter] = random.uniform(-s,s)  

    def single_diamond_square_step(self, d, w, s, avg, frame):
        # w is the dist from one "new" cell to the next
        # v is the dist from a "new" cell to the nbs to average over

        global counter
        n = d.shape[0]
        v = w//2
                
        # offsets:
        diamond = [ (-1,-1), (-1,1), (1,1), (1,-1) ]
        square = [ (-1,0), (0,-1), (1,0), (0,1) ]

        # (i,j) are always the coords of the "new" cell
        # Diamond Step
        for i in range( v, n, w ):
            for j in range( v, n, w ):
                self.update_random_table(s, frame)
                d[i, j] = getattr(self, avg)( d, i, j, v, diamond ) + self.rnd_table[counter]
                counter = counter + 1

        # Square Step, rows
        for i in range( v, n, w ):
            for j in range( 0, n, w ):
                self.update_random_table(s, frame)
                d[i, j] = getattr(self, avg)( d, i, j, v, square ) + self.rnd_table[counter] 
                counter = counter + 1
        
        # Square Step, cols
        for i in range( 0, n, w ):
            for j in range( v, n, w ):
                self.update_random_table(s, frame)
                d[i, j] = getattr(self, avg)( d, i, j, v, square ) + self.rnd_table[counter] 
                counter = counter + 1
            
    def make_terrain(self, n, ds, bdry, frame):
        # Returns an n-by-n landscape using the Diamond-Square algorithm, using
        # roughness delta ds (0..1). bdry is an averaging fct, including the
        # bdry conditions: fixed() or periodic(). n must be 1+2**k, k integer.
        
        global counter

        d = np.zeros( n*n ).reshape( n, n )
        w, s = n-1, 1.0
        counter = 0
        while w > 1:
            self.single_diamond_square_step(d, w, s, bdry, frame)

            w //= 2
            s *= ds

        return d

    def set_palette(self): 
        # Create a colormap  (palette with ordered RGB colors)

        color_table_storm = [] 
        for k in range(0,29):  
            color_table_storm.append([k/28, k/28, k/28])

        color_table_vincent = []
        for k in range(0,29):  
            red  = 0.9*abs(np.sin(0.20*k))  #  0.9 | 0.20
            green= 0.6*abs(np.sin(0.21*k))  #  0.6 | 0.21
            blue = 1.0*abs(np.sin(0.54*k))  #  1.0 | 0.54
            color_table_vincent.append([red, green, blue])

        color_table_terrain = [
                (0.44314, 0.67059, 0.84706),
                (0.47451, 0.69804, 0.87059),
                (0.51765, 0.72549, 0.89020),
                (0.55294, 0.75686, 0.91765),
                (0.58824, 0.78824, 0.94118),
                (0.63137, 0.82353, 0.96863),
                (0.67451, 0.85882, 0.98431),
                (0.72549, 0.89020, 1.00000),
                (0.77647, 0.92549, 1.00000),
                (0.84706, 0.94902, 0.99608),
                (0.67451, 0.81569, 0.64706),
                (0.58039, 0.74902, 0.54510),
                (0.65882, 0.77647, 0.56078),
                (0.74118, 0.80000, 0.58824),
                (0.81961, 0.84314, 0.67059),
                (0.88235, 0.89412, 0.70980),
                (0.93725, 0.92157, 0.75294),
                (0.90980, 0.88235, 0.71373),
                (0.87059, 0.83922, 0.63922),
                (0.82745, 0.79216, 0.61569),
                (0.79216, 0.72549, 0.50980),
                (0.76471, 0.65490, 0.41961),
                (0.72549, 0.59608, 0.35294),
                (0.66667, 0.52941, 0.32549),
                (0.67451, 0.60392, 0.48627),
                (0.72941, 0.68235, 0.60392),
                (0.79216, 0.76471, 0.72157),
                (0.87843, 0.87059, 0.84706),
                (0.96078, 0.95686, 0.94902)
        ]
        if self.palette == 'Storm':
            color_table = color_table_storm
        elif self.palette == 'Terrain':
            color_table = color_table_terrain
        elif self.palette == 'Vincent':
            color_table = color_table_vincent
        return(color_table)

    def morphing(self):
        # create all the images for the video
        # morphing from 'start' to 'end' image
        try:
            size = (self.n - 1) / 64 

            random.seed(self.start_seed)
            frame=0
            start_terrain = self.make_terrain(self.n, self.ds, 
                                              self.bdry, frame)
            random.seed(self.end_seed)
            frame=-1
            end_terrain = self.make_terrain( self.n, self.ds, 
                                            self.bdry, frame)
            progress_bar = st.progress(0, "Morphing..Running Frame: 0")
            if col_morphing:
                col_table_start = np.array(set_palette('Terrain'))**1.00 * np.array(set_palette('Vincent'))**0.50
                col_table_end = np.array(set_palette('Terrain'))**1.50

            for frame in range(0, self.Nframes):
                progress_bar.progress(frame, f"Morphing..Running Frame: {frame}")
                A = frame/(self.Nframes - 1)
                B = 1 - A
                tmp_terrain = B * start_terrain + A * end_terrain
                if col_morphing:    # both palettes must have same size
                    tmp_col_table = col_table_start**B * col_table_end**A
                    tmp_cm = \
                        matplotlib.colors.LinearSegmentedColormap.from_list('temp',tmp_col_table)
                else:
                    tmp_cm = cm
                
                # filename of image in current frame                
                image = \
                    self.image_dir + self.image_file_prefix + str(frame) + self.image_file_suffix
                    
                # create n-by-n pixel fig                    
                plt.figure( figsize=(size, size), dpi=self.dpi ) 
                plt.tick_params( left=False, bottom=False,
                                labelleft=False, labelbottom=False )
                plt.imshow( tmp_terrain, cmap=tmp_cm )
                
                # Save to file            
                plt.savefig(image,bbox_inches='tight',
                            pad_inches=0,dpi=self.dpi)  
                plt.close()
                self.flist.append(image)   
        except Exception as e:
            logging.error(f"Error in {module}: {str(e)}")
            
    def evolution(self):
        try:
            module = "Evolution"
            width_perc = constants.WIDTH_RESIZE_PERC
            side_perc = (100 - width_perc)/2    
            _, center_container, _ = st.columns([side_perc, 
                                                 width_perc, 
                                                 side_perc])
            step = "Defining Progress Bar"            
            progress_bar = center_container.progress(0, "Evolution..Running Frame: 0")
            color_table = self.set_palette()
            print(color_table)
            cm = matplotlib.colors.LinearSegmentedColormap.from_list('geo-smooth', color_table)            
            # create all the images for the video
            random.seed(self.start_seed)
            step = "Beginning Loop"
            
            image_placeholder = center_container.empty()
            for frame in range(0, self.Nframes):
                step = f"Loop {frame}"
                progress_bar.progress(int((frame+1)/self.Nframes*100), f"Evolution..Running Frame {frame+1}")
                
                # filename of image in current frame
                step = f"Loop {frame+1}, Getting image"
                image = \
                    f"{self.image_dir}/{self.image_file_prefix}{frame}{self.image_file_suffix}"
                step = f"Loop {frame}, Image: {image}" 
                size = (self.n - 1) / 64 
                step = f"Loop {frame}, Setting figure"

                # create n-by-n pixel fig                
                plt.figure( figsize=(size, size), dpi=self.dpi )
                step = f"Loop {frame}, Setting tick params"
                plt.tick_params( left=False, 
                                bottom=False, 
                                labelleft=False, 
                                labelbottom=False )
                step = f"Loop {frame}, Calling Terrain"
                
                terrain = self.make_terrain(self.n, 
                                            self.ds, 
                                            self.bdry, 
                                            frame )
                # print(terrain.shape)
                step = f"Loop {frame}, Image show"
                plt.imshow(terrain, cmap=cm )
                
                step = f"Loop {frame}, Saving image {image} to file"
                
                # Save to file          
                plt.savefig(image,
                            bbox_inches='tight',
                            pad_inches=0,
                            dpi=self.dpi)
                
                plt.close()
                step = f"Loop {frame}, Showing image {image}"
                image_placeholder.image(image)
                self.flist.append(image)
            logging.info(f"Image List: {self.flist}")
        except Exception as e:
            logging.error(f"Module: {module}, Step:{step}, Message: {str(e)}")
            # st.error(f"Current Dir: {os.getcwd()}")
        finally:
            progress_bar.empty()
            image_placeholder.empty()
                    
    def run(self):
        if self.method == "Evolution":
            self.evolution()
        else:
            self.morphing()