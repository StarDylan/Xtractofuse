from open3d import *    

def main():
    cloud = io.read_point_cloud("output.ply") 
    visualization.draw_geometries([cloud])   

if __name__ == "__main__":
    main()