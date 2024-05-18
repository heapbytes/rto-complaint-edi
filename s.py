from flask import render_template

@app.route('/run_face_detection')
def run_face_detection():
    # Your face detection logic here
    detected_face_name = "John Doe"  # Example name, replace with actual detected name
    return render_template('index.html', detected_face_name=detected_face_name)


