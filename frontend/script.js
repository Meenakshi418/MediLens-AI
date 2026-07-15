console.log("JavaScript connected!");

// getting refrences
const pdfInput = document.getElementById("pdfInput");

const uploadBTN = document.getElementById("uploadBTN");

const questionInput = document.getElementById("questionInput");

const askBTN = document.getElementById("askBTN");

const answerText = document.getElementById("answerText");

const imageContainer = document.getElementById("imageContainer");

//button clicks
uploadBTN.addEventListener("click" , async function (){

    // files list
    const files = pdfInput.files;
    console.log(files);

    // send data as form data instead of json. as browser sends formdata
    const formData = new FormData();
    for (const file of files){
        formData.append("files" , file);
    }
    // send to backend(python server)
    const response = await fetch("http://127.0.0.1:8000/upload", 
        {
            method: "POST",
            body: formData
    });
    // read backend response
    const data = await response.json();
    console.log(data);
    alert(data.message);
});


askBTN.addEventListener( "click" , async function () {
    const question = questionInput.value;

    if (question.trim() === "") {
        alert("Enter a question first.");
        return;
    }

    answerText.innerText = "Thinking...";
    
    const response = await fetch ("http://127.0.0.1:8000/ask",{
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question
        })
    });

    if (!response.ok) {
        alert("Backend error");
        return;
    }

    const data = await response.json();

    answerText.innerText = data.answer;

    imageContainer.innerHTML = "";

    for (const img of data.images) {

        const image = document.createElement("img");

        image.src = img;

        image.width = 250;

        imageContainer.appendChild(image);

    }
});

// new 
const fileName=document.getElementById("fileName");

pdfInput.addEventListener("change",()=>{

if(pdfInput.files.length===0){

fileName.textContent="No files selected";

}
else{

fileName.textContent=
pdfInput.files.length+" file(s) selected";

}

});