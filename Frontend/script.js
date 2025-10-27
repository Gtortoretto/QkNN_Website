// --- Constants ---
const BACKEND_URL = 'https://gtortoretto.pythonanywhere.com/classify';

const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const classifyBtn = document.getElementById('classify-btn');
const clearBtn = document.getElementById('clear-btn');
const resultSpan = document.getElementById('result');
const loader = document.getElementById('loader');

let drawing = false;

// --- Drawing Logic ---
function startPosition(e) {
    drawing = true;
    draw(e);
}

function endPosition() {
    drawing = false;
    ctx.beginPath(); // Reset the path
}

function draw(e) {
    if (!drawing) return;

    // Make drawing thicker and smoother
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    // Get correct mouse position relative to the canvas
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// --- Button Functions ---
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultSpan.innerText = '-';
}

async function classifyDrawing() {
    // 1. Create a smaller in-memory canvas to resize the drawing to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // 2. Extract the pixel data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = [];
    // We only need the alpha (transparency) channel since we drew in black
    for (let i = 3; i < imageData.data.length; i += 4) {
        pixels.push(imageData.data[i]);
    }

    // 3. Show feedback and send to backend
    loader.style.display = 'block';
    resultSpan.innerText = '-';
    classifyBtn.disabled = true;

    try {
        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels: pixels }),
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        
        const result = await response.json();
        resultSpan.innerText = result.prediction;

    } catch (error) {
        console.error('Error:', error);
        resultSpan.innerText = 'Error!';
    } finally {
        // 4. Hide feedback when done
        loader.style.display = 'none';
        classifyBtn.disabled = false;
    }
}


// --- Event Listeners ---
canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseleave', endPosition); // Stop drawing if mouse leaves canvas

classifyBtn.addEventListener('click', classifyDrawing);
clearBtn.addEventListener('click', clearCanvas);