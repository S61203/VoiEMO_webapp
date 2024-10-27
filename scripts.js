document.addEventListener('DOMContentLoaded', function () {
    const homePage = document.getElementById('home-page');
    const descriptionPage = document.getElementById('description-page');
    const getStartedPage = document.getElementById('get-started-page');
    const projectDescriptionButton = document.getElementById('project-description');
    const getStartedButton = document.getElementById('get-started');
    const backHomeButton = document.getElementById('back-home');
    const backHomeButton2 = document.getElementById('back-home-2');
    const dragDropArea = document.getElementById('drag-drop-area');
    const fileInput = document.getElementById('file-input');
    const fileOutput = document.getElementById('file-output');

    function showPage(page) {
        homePage.classList.add('hidden');
        descriptionPage.classList.add('hidden');
        getStartedPage.classList.add('hidden');
        page.classList.remove('hidden');
    }

    projectDescriptionButton.addEventListener('click', function () {
        showPage(descriptionPage);
    });

    getStartedButton.addEventListener('click', function () {
        showPage(getStartedPage);
    });

    backHomeButton.addEventListener('click', function () {
        showPage(homePage);
    });

    backHomeButton2.addEventListener('click', function () {
        showPage(homePage);
    });

    dragDropArea.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        const files = fileInput.files;
        if (files.length > 0) {
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('file', files[i]);
            }

            // Display the uploaded files
            fileOutput.innerHTML = '';
            Array.from(files).forEach(file => {
                fileOutput.innerHTML += `<p>${file.name}</p>`;
            });

            // Post the files to the server
            fetch('/predict', {  // Change '/detect' to '/predict'
                method: 'POST',
                body: formData
            })
            .then(response => response.json()) // Parse JSON response
            .then(data => {
                if (data.error) {
                    fileOutput.innerHTML += `<p>Error: ${data.error}</p>`;
                } else {
                    fileOutput.innerHTML += `<p>Detected Emotion: ${data.emotion}</p>`;
                }
            })
            .catch(error => {
                fileOutput.innerHTML += `<p>Unexpected Error: ${error.message}</p>`;
            });
        }
    });
});
