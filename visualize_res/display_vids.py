from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str, required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        # frame_idx = os.path.splitext(p.split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        name = request.args.get('name')
        seed = request.args.get('seed')
        out = """<style>
                th, tr, td {
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
                
        if name is None:
            name = ''
        for vf in glob.glob(f'{args.sample_dir}/{name}/*'):
            out += f"<h2> {vf} </h2>"
            if seed is not None and seed not in vf:
                continue
            for v in glob.glob(f'{vf}/sample[0-9][0-9].mp4'):
                if len(v) == 0:
                    out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
                    continue
                else:
                  print(v)
                  out += f"""
                  <td>
                  <video width=\"512\" height=\"512\" autoplay muted controls loop> 
                      <source src=\"/files/{v}\" type=\"video/mp4\">
                      Your browser does not support the video tag.
                      </video>
                  </td>
                  """
            out += "</tr>"     
            out += "</table>"
            out += "<br> <hr>"
                
        return out

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)