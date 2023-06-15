using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;
using System;
using System.IO;
using System.Linq;

namespace Face_Detection.Controllers
{
    public class FaceController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        public IActionResult DetectFace(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
            {
                return BadRequest("No file selected.");
            }

            using var memoryStream = new MemoryStream();
            imageFile.CopyTo(memoryStream);
            byte[] imageBytes = memoryStream.ToArray();

            using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Color);

            // Define the desired passport photo size
            int passportWidth = (int)(2.0 * 35.0);  // 2 inches at 300 dpi (dots per inch)
            int passportHeight = (int)(2.0 * 45.0); // 2 inches at 300 dpi (dots per inch)

            // Resize the image to the passport size
            Cv2.Resize(mat, mat, new Size(passportWidth, passportHeight));

            // Get the dimensions of the resized image
            int imageWidth = mat.Width;
            int imageHeight = mat.Height;

            using var cascade = new CascadeClassifier();
            cascade.Load("haarcascade_frontalface_default.xml");

            var faces = cascade.DetectMultiScale(mat);

            bool hasFace = faces.Length > 0;
            bool isCentered = false;
            bool isPlainColor = false;
            Scalar backgroundColor = Scalar.Black; // Default background color is black

            if (hasFace)
            {
                // Check if the face is centered
                var imageCenter = new Point2f(imageWidth / 2f, imageHeight / 2f);
                var faceCenter = new Point2f(faces[0].X + (faces[0].Width / 2f), faces[0].Y + (faces[0].Height / 2f));

                var tolerance = 0.1;

                isCentered = Math.Abs(imageCenter.X - faceCenter.X) <= tolerance * imageWidth &&
                             Math.Abs(imageCenter.Y - faceCenter.Y) <= tolerance * imageHeight;

                // Get the average color of the background region
                var backgroundRegion = new Rect(0, 0, imageWidth, imageHeight);
                var backgroundMat = new Mat(mat, backgroundRegion);

                // Convert the backgroundMat to grayscale for color analysis
                var grayMat = new Mat();
                Cv2.CvtColor(backgroundMat, grayMat, ColorConversionCodes.BGR2GRAY);

                // Calculate the histogram of the grayscale image
                var hist = new Mat();
                int[] channels = { 0 };
                int[] histSize = { 256 };
                Rangef[] ranges = { new Rangef(0, 256) };
                Cv2.CalcHist(new[] { grayMat }, channels, null, hist, channels.Length, histSize, ranges);

                // Find the bin with the highest frequency
                Point minLoc, maxLoc;
                double minValue, maxValue;
                Cv2.MinMaxLoc(hist, out minValue, out maxValue, out minLoc, out maxLoc);
                var maxValIdx = maxLoc;

                // Get the average color in the background region
                var backgroundGrayValue = (int)maxValIdx.Y;

                backgroundColor = new Scalar(backgroundGrayValue, backgroundGrayValue, backgroundGrayValue);

                // Check if the background color is within the tolerance range
                isPlainColor = IsPlainColor(backgroundColor, 0.1);
            }

            var result = new
            {
                HasFace = hasFace,
                IsCentered = isCentered,
                IsPlainColor = isPlainColor,
                BackgroundColor = new[] { backgroundColor.Val0, backgroundColor.Val1, backgroundColor.Val2 },
                Faces = faces,
                ImageWidth = imageWidth, // Include the image dimensions in the result
                ImageHeight = imageHeight
            };

            return Json(result);
        }







        private bool IsPlainColor(Scalar color, double tolerance)
        {
            // Get the color values of each channel
            byte[] channels = { (byte)color.Val0, (byte)color.Val1, (byte)color.Val2 };

            // Calculate the difference between channel values
            int diff01 = Math.Abs(channels[0] - channels[1]);
            int diff02 = Math.Abs(channels[0] - channels[2]);
            int diff12 = Math.Abs(channels[1] - channels[2]);

            // Check if the difference is within the tolerance range
            bool isPlainColor = diff01 <= tolerance && diff02 <= tolerance && diff12 <= tolerance &&
                               channels[0] >= tolerance && channels[1] >= tolerance && channels[2] >= tolerance;

            return isPlainColor;
        }


    }
}
