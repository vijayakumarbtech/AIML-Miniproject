🔹 Step 1: Clone the GitHub Repository
Open your terminal or command prompt and run:

git clone https://github.com/<your-fake-image-video-detection-repo>.git
🔁 Replace the link with the actual GitHub URL you are using.

🔹 Step 2: Install Required Libraries
Navigate into the project folder:

cd your-repo-folder
Then install the dependencies:

pip install -r requirements.txt
If there’s no requirements.txt, install manually:

pip install opencv-python scikit-learn numpy scipy scikit-image joblib

🔹 Step 3: Prepare the Datasets
 create two folders:

real/ → Put real images (available in git)

fake/ → Put fake images (available in git)

For video:

Place test videos in a videos/ folder or as needed by the script.

🔹 Step 4: Train the Model
Run the training script (if included):

python  train_model.py
This will create a file like model.pkl which contains the trained model.

🔹 Step 5: Detect a Fake or Real Image
Run the image detection script:

python detect_image.py  test.jpg
Replace test.jpg with the path to your test image.

🔹 Step 6: Detect a Fake or Real Video
Run the video detection script:

python detect_video.py  test_video.mp4
Replace test_video.mp4 with your test video file.

🔹 Step 7: View the Output
The terminal will show:

For images:
✅ "This is a Real Image" or ❌ "This is a Fake Image"

For videos:
✅ Majority voting result like "Video is Fake" or "Video is Real"
