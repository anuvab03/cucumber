document.getElementById('imageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    
    if (file) {
        
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
            document.getElementById('preview').style.display = 'block';
        };
        reader.readAsDataURL(file);

        
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('output').value = "Error: " + data.error;
            } else {
                document.getElementById('output').value = 
                    `Prediction: ${data.prediction}\nConfidence: ${data.confidence}\n\n${data.file_content}`;
            }
        })
        .catch(error => {
            document.getElementById('output').value = "Error processing the image.";
            console.error("Error:", error);
        });
    }
});
