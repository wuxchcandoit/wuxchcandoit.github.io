from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from math import exp, log, sqrt, pi
from fordFulkerson import FordFulkerson
from boykovKolmogorov import BoykovKolmogorov

graph_cut_algo = {"ff": FordFulkerson, "bk": BoykovKolmogorov}
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"
CUTCOLOR = (255, 0, 0)
# SOURCE, SINK = -2, -1

def norm_pdf(x, mu, sigma):
    factor = (1. / (abs(sigma) * sqrt(2 * pi)))
    return factor * exp( -(x-mu)**2 / (2 * sigma**2) )

class MinCutSeg(object):
    def __init__(self, image, sigma_factor=50, lambda_factor=2):
        self.image = image
        self.sigma_factor = sigma_factor
        self.lambda_factor = lambda_factor
        self.h, self.w = image.shape
        self.s = self.h * self.w
        self.t = self.s + 1
        self.tuple_to_edge = dict()
        self.obj_seeds = []
        self.bkg_seeds = []
        self.pixel_values = {p: image[p] for p in self.pixels()}
        self.calculate_boundary_costs()

    def segmentation(self, algo):
        seeds, image_with_seeds = self.plant_seeds()

        cv2.imwrite("./image_with_seeds.jpg", image_with_seeds)
        cv2.imwrite("./seeds.jpg", seeds * 127)

        # Update regional penalt
        obj_mean, obj_std = self.calculate_normal(self.obj_seeds)
        bkg_mean, bkg_std = self.calculate_normal(self.bkg_seeds)
        self.regional_penalty_bkg = {p: 0 if p in self.obj_seeds else self.k_factor if p in self.bkg_seeds else self.regional_cost(p, obj_mean, obj_std) for p in self.pixels()}
        self.regional_penalty_obj = {p: self.k_factor if p in self.obj_seeds else 0 if p in self.bkg_seeds else self.regional_cost(p, bkg_mean, bkg_std) for p in self.pixels()}
        # Update the graph, mainly t-links
        self.update_graph()
        if algo == "ap":
            cut = graph_cut_algo[algo](self.tuple_to_edge, self.s, self.t, self.h, self.w)
        elif algo == "bk":
            bk = graph_cut_algo[algo](self.tuple_to_edge, self.s, self.t, self.h, self.w)
            cut, boundary = bk.max_flow()
        return cut
            # cut = bk.max_flow()
        # return cut
    
    def update_graph(self):
        # Boundary costs
        # for x in range(0, self.h, 2):
        #     for y in range(0, self.w, 2):
        for x in range(0, self.h):
            for y in range(0, self.w):
                p = (x, y)
                for n_p in self.special_neighbours(*p):
                    u = p[0] * self.w + p[1]
                    v = n_p[0] * self.w + n_p[1]
                    self.tuple_to_edge[(u, v)] = self.tuple_to_edge[(v, u)] = self.boundary_costs[p][n_p]

        # Regional costs
        for p in self.pixels():
            x, y = p[0], p[1]
            v = x * self.w + y
            self.tuple_to_edge[(self.s, v)] = self.regional_penalty_obj[p]
            self.tuple_to_edge[(v, self.t)] = self.regional_penalty_bkg[p]
        
    def calculate_normal(self, points):
        values = [self.pixel_values[p] for p in points]
        return np.mean(values), max(np.std(values), 0.00001)

    def regional_cost(self, point, mean, std):
        prob = max(norm_pdf(self.pixel_values[point], mean, std), 0.000000000001)
        return - self.lambda_factor * log(prob)

    def pixels(self):
        for x in range(self.h):
            for y in range(self.w):
                yield(x, y)

    # def special_neighbours(self, x, y):
    #     return ((i, j) for (i, j) in [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1)] if 0 <= i < self.h and 0 <= j < self.w and (i != x or j != y))
    def special_neighbours(self, x, y):
        return ((i, j) for (i, j) in [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1)] if 0 <= i < self.h and 0 <= j < self.w and (i != x or j != y))

    def boundary_penalty(self, p_a, p_b):
        if(self.pixel_values[p_a] > self.pixel_values[p_b]):
            i_delta = self.pixel_values[p_a] - self.pixel_values[p_b]
        else:
            i_delta = self.pixel_values[p_b] - self.pixel_values[p_a]
        distance = abs(p_a[0] - p_b[0]) + abs(p_a[1] - p_b[1])
        return 30* exp(- i_delta**2 / (2 * self.sigma_factor**2)) / distance

    def calculate_boundary_costs(self):
        self.boundary_costs = {}
        for p in self.pixels():
            self.boundary_costs[p] = {}
            for n_p in self.special_neighbours(*p):
                self.boundary_costs[p][n_p] = self.boundary_penalty(p, n_p)
        self.k_factor = 1. + max(sum(self.boundary_costs[p].values()) for p in self.pixels())


    def plant_seeds(self):
        def draw_lines(x, y, pixel_type):
            if(pixel_type == OBJ):
                color, code = OBJCOLOR, OBJCODE
            else:
                color, code = BKGCOLOR, BKGCODE
            cv2.circle(image, (x, y), radius=5, color=color, thickness=-1)
            cv2.circle(seeds, (x, y), radius=5, color=code, thickness=-1)
        
        def onMouse(event, x, y, flags, pixel_type):
            global drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                draw_lines(x, y, pixel_type)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                draw_lines(x, y, pixel_type)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
        
        def paint_seeds(pixel_type):
            print("Planting ", pixel_type, " seeds")
            global drawing
            drawing = False
            window_name = "Plant " + pixel_type + " seeds"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, onMouse, pixel_type)
            while(1):
                cv2.imshow(window_name, image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()

        seeds = np.zeros(self.image.shape, dtype='uint8')
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        global drawing
        drawing = False

        paint_seeds(OBJ)
        paint_seeds(BKG)

        for x in range(self.h):
            for y in range(self.w):
                if(seeds[x, y] == OBJCODE):
                    self.obj_seeds.append((x, y))
                elif(seeds[x, y] == BKGCODE):
                    self.bkg_seeds.append((x, y))
        return seeds, image


def display_cut_bk(image, cut, obj_seeds, bkg_seeds, s, t):
    h, w = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(len(cut))
    for point in cut:
        if(point[1] == t):
            image[point[0] // w, point[0] % w] = [0, 0, 0] 
        elif point[0] == s:
            image[point[1] // w, point[1] % w] = [255, 255, 255]
    for seed in obj_seeds:
        image[seed] = [255, 255, 255]
    for seed in bkg_seeds:
        image[seed] = [0, 0, 0]
    return image

def display_cut_ff(image, cut, obj_seeds, bkg_seeds, s, t):
    h, w = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(len(cut))
    for point in cut:
        if(point[1] == t):
            image[point[0] // w, point[0] % w] = [255, 255, 255]
        elif point[0] == s:
            image[point[1] // w, point[1] % w] = [0, 0, 0]
    for seed in obj_seeds:
        image[seed] = [255, 255, 255]
    for seed in bkg_seeds:
        image[seed] = [0, 0, 0]
    return image

def show_image(image):
    window_name = "Segmentation Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main(image_file, algo="ff", size=(200, 200)): 
    path_name = os.path.splitext(image_file)[0]
    # Loads image in grayscale mode
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) 
    # image = cv2.resize(image, size) # think resize
    img = MinCutSeg(image)
    cut = img.segmentation(algo)
    # print(cut)
    if algo == "bk":
        image = display_cut_bk(image, cut, img.obj_seeds, img.bkg_seeds, img.s, img.t)
    elif algo == "ff":
        image = display_cut_ff(image, cut, img.obj_seeds, img.bkg_seeds, img.s, img.t)        
    show_image(image)
    save_name = path_name + 'cut.jpg'
    cv2.imwrite(save_name, image)
    print("Saved image as ", save_name)
 
def parseArgs():
    def algorithm(string):
        if string in graph_cut_algo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graph_cut_algo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument("--algo", "-a", default="bk", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    main(args.image_file, args.algo) 
    