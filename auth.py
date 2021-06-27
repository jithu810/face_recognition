from flask import Flask,render_template,make_response,url_for,request
import random
from newmod import TrackImages,Take

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/verify')
def verify():
    return render_template("verify.html")

@app.route('/videos', methods=["POST"])
def videos():
    video_file = request.files["file"]
    file_name = "test/" + str(random.randint(0, 100000))+".mp4"
    video_file.save(file_name)
    TrackImages(file_name)
    Take(file_name)
    return "verified"




if __name__ == '__main__':
   app.run()
