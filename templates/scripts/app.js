// Example to show prediction result on submit
document.getElementById('studentForm').addEventListener('submit', function(e) {
    e.preventDefault();
    document.getElementById('predictionResult').textContent = "THE prediction is (your result here)";
    // Replace above line with actual prediction logic!
});
