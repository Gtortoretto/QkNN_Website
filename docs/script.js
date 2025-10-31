document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Constants and Global State ---
    const BACKEND_URLS = {
        pythonanywhere: 'https://gtortoretto.pythonanywhere.com/classify',
        localhost: 'http://127.0.0.1:5000/classify'
    };
    let currentBackendUrl = BACKEND_URLS.pythonanywhere; // Default URL
    const MOBILE_BREAKPOINT = 820;
    let drawing = false;
    let historyLog = []; 
    const HISTORY_STORAGE_KEY = 'mnist-history-log'; 
    let lastResultState = null;
    
    // Drawing configuration
    let drawingConfig = {
        strokeWidth: 20,
        fountainPen: false
    };
    let lastX = 0;
    let lastY = 0;
    let lastTime = 0;
    let lastWidth = 20; // Track last width for smooth transitions
    let velocityHistory = []; // Keep a history of recent velocities for smoothing

    let algorithmConfigs = {
        'qknn_sim': {
            training_size: 2500,
            pca_components: 16,
            k: 5,
            shots: 1024,
            method: 'default'
        },
        'knn_classical': {
            training_size: 25000,
            pca_components: 200,
            k: 5,
            method: 'default'
        }
    };

    const ALGORITHM_DISPLAY_NAMES = {
        knn_classical: 'kNN',
        qknn_sim: 'QkNN',
        qknn_real: 'rQkNN'
    };

    // --- 2. DOM Element Selection ---
    const body = document.body;
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const classifyBtn = document.getElementById('classify-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultDisplay = document.getElementById('result-display');
    const loader = document.getElementById('loader');
    const navbar = document.querySelector('.navbar');
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const settingsCog = document.getElementById('settingsCog');
    const settingsMenu = document.getElementById('settingsMenu');
    const historyTableBody = document.getElementById('historyTableBody');
    const historyTableHead = document.querySelector('#historyContainer table thead tr');
    const accuracyTally = document.getElementById('accuracyTally');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    
    // Settings Panel Elements
    const backendRadios = document.querySelectorAll('input[name="backend"]');
    const algorithmRadios = document.querySelectorAll('input[name="algorithm"]');
    const showTimeToggle = document.getElementById('showTimeToggle');
    const historyToggle = document.getElementById('historyToggle');
    const historyContainer = document.getElementById('historyContainer');
    const preprocessToggle = document.getElementById('preprocessToggle');
    const preprocessSettingsCog = document.getElementById('preprocessSettingsCog');
    const preprocessSubMenu = document.getElementById('preprocessSubMenu');
    const comToggle = document.getElementById('comToggle');
    const blurToggle = document.getElementById('blurToggle');
    const blurSlider = document.getElementById('blurSlider');
    const blurValue = document.getElementById('blurValue');
    const thinningToggle = document.getElementById('thinningToggle');
    const thinningSlider = document.getElementById('thinningSlider');
    const thinningValue = document.getElementById('thinningValue');
    
    // Debug view elements
    const debugViewToggle = document.getElementById('debugViewToggle');
    const debugArea = document.getElementById('debug-area');
    const debugCanvasUser = document.getElementById('debug-canvas-user');
    const debugCanvasMnist = document.getElementById('debug-canvas-mnist');
    const mnistLabel = document.getElementById('mnist-label');
    const mnistDigitFilter = document.getElementById('mnist-digit-filter');
    const compareMnistBtn = document.getElementById('compare-mnist-btn');

    // Algorithm configuration elements
    const qknnSettingsCog = document.getElementById('qknnSettingsCog');
    const qknnSubMenu = document.getElementById('qknnSubMenu');
    const qknnTrainingSize = document.getElementById('qknnTrainingSize');
    const qknnTrainingSizeValue = document.getElementById('qknnTrainingSizeValue');
    const qknnPcaComponents = document.getElementById('qknnPcaComponents');
    const qknnPcaComponentsValue = document.getElementById('qknnPcaComponentsValue');
    const qknnK = document.getElementById('qknnK');
    const qknnKValue = document.getElementById('qknnKValue');
    const qknnShots = document.getElementById('qknnShots');
    const qknnShotsValue = document.getElementById('qknnShotsValue');

    const knnSettingsCog = document.getElementById('knnSettingsCog');
    const knnSubMenu = document.getElementById('knnSubMenu');
    const knnTrainingSize = document.getElementById('knnTrainingSize');
    const knnTrainingSizeValue = document.getElementById('knnTrainingSizeValue');
    const knnPcaComponents = document.getElementById('knnPcaComponents');
    const knnPcaComponentsValue = document.getElementById('knnPcaComponentsValue');
    const knnK = document.getElementById('knnK');
    const knnKValue = document.getElementById('knnKValue');

    // Pencil configuration elements
    const pencilCog = document.getElementById('pencilCog');
    const pencilMenu = document.getElementById('pencilMenu');
    const fountainPenToggle = document.getElementById('fountainPenToggle');
    const strokeWidthSlider = document.getElementById('strokeWidth');
    const strokeWidthValue = document.getElementById('strokeWidthValue');


    // --- 3. Functions ---

    // --- Settings Logic ---
    function saveSettings() {
        const selectedBackend = document.querySelector('input[name="backend"]:checked').value;
        const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked').value;
        
        localStorage.setItem('settings_backend', selectedBackend);
        localStorage.setItem('settings_algorithm', selectedAlgorithm);
        localStorage.setItem('settings_showTime', showTimeToggle.checked);
    }

    function loadSettings() {
        const savedBackend = localStorage.getItem('settings_backend') || 'pythonanywhere';
        document.querySelector(`input[name="backend"][value="${savedBackend}"]`).checked = true;

        const savedAlgorithm = localStorage.getItem('settings_algorithm') || 'qknn_sim';
        document.querySelector(`input[name="algorithm"][value="${savedAlgorithm}"]`).checked = true;
        
        const savedShowTime = localStorage.getItem('settings_showTime') === 'true';
        showTimeToggle.checked = savedShowTime;

        historyToggle.checked = false;
        historyContainer.classList.remove('active');
    }

    function saveHistory() {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(historyLog));
    }

    function loadHistory() {
        const savedHistory = localStorage.getItem(HISTORY_STORAGE_KEY);
        if (savedHistory) {
            historyLog = JSON.parse(savedHistory);
        }
    }

    function renderHistoryTable() {
        // 1. Clear any existing content to prevent duplicates
        historyTableHead.innerHTML = '';
        historyTableBody.innerHTML = '';

        // 2. Discover all unique algorithm names
        const allAlgorithmKeys = new Set();
        historyLog.forEach(entry => {
            Object.keys(entry.predictions).forEach(key => allAlgorithmKeys.add(key));
        });
        const algorithmColumns = Array.from(allAlgorithmKeys);

        // 3. Create the dynamic table headers
        let headersHTML = `
            <th>Drawing</th>
            <th>Correct Label</th>
            <th>Duration (ms)</th>
        `;
        algorithmColumns.forEach(key => {
            headersHTML += `<th>${key.replace(/_/g, ' ')}</th>`;
        });
        headersHTML += `<th></th>`; // <-- ADD THIS LINE for the delete button column
        historyTableHead.innerHTML = headersHTML;

        // 4. Create and append a row for each entry
        historyLog.forEach(entry => {
            const row = document.createElement('tr');
            
            let rowHTML = `
                <td><img src="${entry.imageDataUrl}" alt="User drawing"></td>
                <td><input type="number" class="correct-label-input" data-id="${entry.id}" min="0" max="9" value="${entry.correctLabel || ''}"></td>
                <td>${entry.duration}</td>
            `;
            
            algorithmColumns.forEach(key => {
                const prediction = entry.predictions[key] || '-';
                rowHTML += `<td>${prediction}</td>`;
            });

            // --- ADD THIS CELL FOR THE DELETE BUTTON ---
            rowHTML += `<td><button class="delete-row-btn" data-id="${entry.id}">&times;</button></td>`;

            row.innerHTML = rowHTML;
            historyTableBody.appendChild(row);
        });
        
        updateAccuracyTally();
    }

    function updateAccuracyTally() {
        const scores = {}; // e.g., { knn_classical: { correct: 5, total: 6 } }

        // 1. Tally up the correct and total predictions for each algorithm
        historyLog.forEach(entry => {
            // Only count entries where the user has provided a correct label
            if (entry.correctLabel !== null && entry.correctLabel !== '') {
                Object.keys(entry.predictions).forEach(algoKey => {
                    // Initialize the score object for a new algorithm if it doesn't exist
                    if (!scores[algoKey]) {
                        scores[algoKey] = { correct: 0, total: 0 };
                    }
                    
                    scores[algoKey].total++;
                    // Check if the prediction matches the user-provided label (use == for type coercion)
                    if (entry.predictions[algoKey] == entry.correctLabel) {
                        scores[algoKey].correct++;
                    }
                });
            }
        });

        // 2. Build the HTML string to display the scores
        let tallyHTML = '';
        const sortedKeys = Object.keys(scores).sort(); // Sort keys for consistent order

        sortedKeys.forEach((key, index) => {
            const score = scores[key];
            const accuracy = (score.total > 0) ? (score.correct / score.total) * 100 : 0;
            
            // Add a separator before all items except the first one
            if (index > 0) {
                tallyHTML += `<span class="separator">&nbsp;|&nbsp;</span>`;
            }
            
            const displayName = key.replace(/_/g, ' '); // Make the name readable
            tallyHTML += `<span>${displayName}: <strong>${accuracy.toFixed(0)}%</strong> (${score.correct}/${score.total})</span>`;
        });
        
        // 3. Update the DOM
        accuracyTally.innerHTML = tallyHTML || 'Enter correct labels to see accuracy.';
    }

    // --- Dark Mode ---
    function setTheme(isDark) {
        if (isDark) {
            body.classList.add('dark-mode');
            darkModeToggle.checked = true;
        } else {
            body.classList.remove('dark-mode');
            darkModeToggle.checked = false;
        }
    }

    // --- Drawing Logic ---
    function getPointerPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return { x: clientX - rect.left, y: clientY - rect.top };
    }

    function calculateStrokeWidth(dx, dy, dt) {
        if (!drawingConfig.fountainPen) {
            return drawingConfig.strokeWidth;
        }
        
        // Calculate velocity (distance / time)
        const distance = Math.sqrt(dx * dx + dy * dy);
        const velocity = dt > 0 ? distance / dt : 0;
        
        // Add to velocity history for smoothing (keep last 3 samples)
        velocityHistory.push(velocity);
        if (velocityHistory.length > 3) {
            velocityHistory.shift();
        }
        
        // Use average velocity for smoother transitions
        const avgVelocity = velocityHistory.reduce((sum, v) => sum + v, 0) / velocityHistory.length;
        
        // Map velocity to stroke width - only gets thinner, never thicker
        // Slow = base width (from slider), fast = thinner
        const minWidth = drawingConfig.strokeWidth * 0.4;  // Thinnest (fast writing)
        const maxWidth = drawingConfig.strokeWidth;         // Thickest = base width (slow writing)
        
        // Normalize velocity with higher threshold for more dramatic effect
        // Typical fast writing: 2-8 pixels per ms
        const normalizedVelocity = Math.min(avgVelocity / 5, 1);
        
        // Direct relationship with smooth curve: high velocity = thin line
        const targetWidth = maxWidth - (Math.pow(normalizedVelocity, 0.7) * (maxWidth - minWidth));
        
        // Interpolate between last width and target width for ultra-smooth transitions
        const smoothingFactor = 0.3; // Lower = smoother but more lag, higher = more responsive
        const width = lastWidth + (targetWidth - lastWidth) * smoothingFactor;
        
        lastWidth = width;
        return width;
    }

    function startPosition(e) {
        e.preventDefault();
        drawing = true;
        const { x, y } = getPointerPos(e);
        lastX = x;
        lastY = y;
        lastTime = Date.now();
        lastWidth = drawingConfig.strokeWidth;
        velocityHistory = []; // Reset velocity history on new stroke
        
        ctx.beginPath();
        ctx.moveTo(x, y);
        
        // Draw a dot when clicking (not dragging)
        ctx.lineWidth = drawingConfig.strokeWidth;
        ctx.lineCap = 'round';
        ctx.strokeStyle = body.classList.contains('dark-mode') ? 'white' : 'black';
        ctx.lineTo(x + 0.1, y + 0.1); // Tiny line to create a dot
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function endPosition() {
        drawing = false;
        ctx.beginPath();
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();
        
        const { x, y } = getPointerPos(e);
        const currentTime = Date.now();
        
        const dx = x - lastX;
        const dy = y - lastY;
        const dt = currentTime - lastTime;
        
        const width = calculateStrokeWidth(dx, dy, dt);
        
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.strokeStyle = body.classList.contains('dark-mode') ? 'white' : 'black';
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
        
        lastX = x;
        lastY = y;
        lastTime = currentTime;
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resetResultDisplay();
        loader.style.display = 'none';
        lastResultState = null;
    }

    function resetResultDisplay() {
        if (!resultDisplay) return;
        resultDisplay.replaceChildren();
        resultDisplay.classList.remove('active', 'single-mode', 'multi-mode', 'with-time', 'without-time');
    }

    function renderResultMessage(message) {
        if (!resultDisplay) return;
        resetResultDisplay();
        const messageEl = document.createElement('div');
        messageEl.className = 'result-message';
        messageEl.textContent = message;
        resultDisplay.appendChild(messageEl);
        resultDisplay.classList.add('active');
    }

    function getDisplayNameForAlgorithm(key) {
        return ALGORITHM_DISPLAY_NAMES[key] || key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
    }

    function formatTimingValue(timeMs) {
        if (typeof timeMs === 'number' && timeMs >= 0) {
            return `${timeMs} ms`;
        }
        return 'N/A';
    }

    function renderSingleResult(row, includeTiming) {
        if (!resultDisplay) return;
        resetResultDisplay();
        const wrapper = document.createElement('div');
        wrapper.className = 'single-result';

        const valueSpan = document.createElement('span');
        valueSpan.className = 'single-value';
        valueSpan.textContent = row.prediction ?? 'N/A';
        wrapper.appendChild(valueSpan);

        if (includeTiming) {
            const timeSpan = document.createElement('span');
            timeSpan.className = 'single-time';
            timeSpan.textContent = `(${formatTimingValue(row.time)})`;
            wrapper.appendChild(timeSpan);
        }

        resultDisplay.appendChild(wrapper);
        resultDisplay.classList.add('active', 'single-mode');
        resultDisplay.classList.add(includeTiming ? 'with-time' : 'without-time');
    }

    function renderMultiResult(rows, includeTiming) {
        if (!resultDisplay) return;
        resetResultDisplay();
        const table = document.createElement('table');

        rows.forEach(row => {
            const tr = document.createElement('tr');

            const nameTd = document.createElement('td');
            nameTd.className = 'algo-name';
            nameTd.textContent = `${row.displayName}:`;
            tr.appendChild(nameTd);

            const predTd = document.createElement('td');
            predTd.className = 'algo-pred';
            predTd.textContent = row.prediction ?? 'N/A';
            tr.appendChild(predTd);

            if (includeTiming) {
                const timeTd = document.createElement('td');
                timeTd.className = 'algo-time';
                timeTd.textContent = `(${formatTimingValue(row.time)})`;
                tr.appendChild(timeTd);
            }

            table.appendChild(tr);
        });

        resultDisplay.appendChild(table);
        resultDisplay.classList.add('active', 'multi-mode');
        resultDisplay.classList.add(includeTiming ? 'with-time' : 'without-time');
    }

    function replayLastResult() {
        if (!lastResultState) {
            resetResultDisplay();
            return;
        }

        const includeTiming = showTimeToggle ? showTimeToggle.checked : true;

        if (lastResultState.type === 'single' && lastResultState.rows?.length) {
            renderSingleResult(lastResultState.rows[0], includeTiming);
        } else if (lastResultState.type === 'multi' && lastResultState.rows?.length) {
            renderMultiResult(lastResultState.rows, includeTiming);
        } else if (lastResultState.type === 'message') {
            renderResultMessage(lastResultState.message);
        }
    }

    // --- START: NEW CODE BLOCK ---
    // The following functions implement the live status dots and improved settings logic.

    function applyBackendFromRadios() {
        const selected = document.querySelector('input[name="backend"]:checked')?.value;
        if (selected && BACKEND_URLS[selected]) {
            currentBackendUrl = BACKEND_URLS[selected];
            updateAlgorithmConfigAvailability(selected);
        }
    }

    function updateAlgorithmConfigAvailability(backend) {
        const isPythonAnywhere = backend === 'pythonanywhere';
        
        // Disable/enable algorithm configuration cog buttons
        if (qknnSettingsCog) {
            qknnSettingsCog.disabled = isPythonAnywhere;
            qknnSettingsCog.style.opacity = isPythonAnywhere ? '0.5' : '1';
            qknnSettingsCog.style.cursor = isPythonAnywhere ? 'not-allowed' : 'pointer';
            qknnSettingsCog.title = isPythonAnywhere ? 'Configuration locked for PythonAnywhere (250 training, 16 PCA)' : 'Configure QkNN settings';
        }
        
        if (knnSettingsCog) {
            knnSettingsCog.disabled = isPythonAnywhere;
            knnSettingsCog.style.opacity = isPythonAnywhere ? '0.5' : '1';
            knnSettingsCog.style.cursor = isPythonAnywhere ? 'not-allowed' : 'pointer';
            knnSettingsCog.title = isPythonAnywhere ? 'Configuration locked for PythonAnywhere (250 training, 16 PCA)' : 'Configure kNN settings';
        }
        
        // Close any open configuration menus when switching to PythonAnywhere
        if (isPythonAnywhere) {
            closeQknnPopover();
            closeKnnPopover();
        }
        
        // Reset to default PythonAnywhere values if switching to it
        if (isPythonAnywhere) {
            algorithmConfigs.qknn_sim.training_size = 250;
            algorithmConfigs.qknn_sim.pca_components = 16;
            algorithmConfigs.knn_classical.training_size = 250;
            algorithmConfigs.knn_classical.pca_components = 16;
        }
    }

    function getSelectedAlgorithm() {
        return document.querySelector('input[name="algorithm"]:checked')?.value || 'qknn_sim';
    }

    async function ping(url, ms = 2000) {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), ms);
        try {
            // mode: 'no-cors' is a trick to check for reachability without needing a full CORS-compliant response.
            await fetch(url.replace('/classify', '/status'), { method: 'GET', mode: 'no-cors', signal: ctrl.signal });
            return true; // If we got any response (no network error), consider it reachable.
        } catch {
            return false;
        } finally {
            clearTimeout(t);
        }
    }

    async function updateBackendStatusDots() {
        // --- PythonAnywhere Check ---
        const paRadio = document.querySelector('input[name="backend"][value="pythonanywhere"]');
        const paDot = document.getElementById('paStatus');
        if (paRadio && paDot) {
            const isOnline = await ping(BACKEND_URLS.pythonanywhere);
            paDot.classList.toggle('green', isOnline);
            paDot.classList.toggle('red', !isOnline);
            paDot.title = isOnline ? 'Online' : 'Offline';
            paRadio.disabled = !isOnline; // Disable radio button if offline
        }

        // --- Localhost Check ---
        const localRadio = document.querySelector('input[name="backend"][value="localhost"]');
        const localDot = document.getElementById('localStatus');
        if (localRadio && localDot) {
            const isOnline = await ping(BACKEND_URLS.localhost);
            localDot.classList.toggle('green', isOnline);
            localDot.classList.toggle('red', !isOnline);
            localDot.title = isOnline ? 'Online' : 'Offline';
            localRadio.disabled = !isOnline; // Disable radio button if offline
        }
        
        // After updating statuses, check if the main button should be disabled
        updateClassifyButtonState();
        
        // Update algorithm config availability based on selected backend
        const selectedBackend = document.querySelector('input[name="backend"]:checked')?.value;
        if (selectedBackend) {
            updateAlgorithmConfigAvailability(selectedBackend);
        }
    }

    function updateClassifyButtonState() {
        const selectedRadio = document.querySelector('input[name="backend"]:checked');
        // Disable the classify button if the currently selected backend radio is disabled
        if (selectedRadio) {
            classifyBtn.disabled = selectedRadio.disabled;
        }
    }

    /**
     * A flexible preprocessing function that applies different steps based on options.
     * @param {object} options - Configuration object for preprocessing steps.
     * @returns {HTMLCanvasElement} A new 28x28 canvas with the processed image.
     */
    function preprocessCanvas(options) {
        let currentCanvas = document.createElement('canvas');
        currentCanvas.width = 280;
        currentCanvas.height = 280;
        currentCanvas.getContext('2d').drawImage(canvas, 0, 0);

        // Step 1: Center of Mass (Conditional)
        if (options.useCoM) {
            const sourceCtx = currentCanvas.getContext('2d', { willReadFrequently: true });
            const imageData = sourceCtx.getImageData(0, 0, currentCanvas.width, currentCanvas.height);
            let sumX = 0, sumY = 0, totalMass = 0;

            for (let y = 0; y < currentCanvas.height; y++) {
                for (let x = 0; x < currentCanvas.width; x++) {
                    const alpha = imageData.data[(y * currentCanvas.width + x) * 4 + 3];
                    if (alpha > 0) {
                        sumX += x * alpha; sumY += y * alpha; totalMass += alpha;
                    }
                }
            }
            
            if (totalMass > 0) {
                const comX = sumX / totalMass;
                const comY = sumY / totalMass;
                const shiftX = currentCanvas.width / 2 - comX;
                const shiftY = currentCanvas.height / 2 - comY;
                
                const shiftedCtx = document.createElement('canvas').getContext('2d');
                shiftedCtx.canvas.width = 280;
                shiftedCtx.canvas.height = 280;
                shiftedCtx.drawImage(currentCanvas, shiftX, shiftY);
                currentCanvas = shiftedCtx.canvas;
            }
        }
        
        // Create final 28x28 canvas
        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = 28;
        finalCanvas.height = 28;
        const finalCtx = finalCanvas.getContext('2d');
        finalCtx.drawImage(currentCanvas, 0, 0, 28, 28);

        // Step 2: Gaussian Blur (Conditional)
        if (options.useBlur) {
            finalCtx.filter = `blur(${options.blurAmount}px)`;
            // Drawing twice can enhance the effect after a blur
            finalCtx.drawImage(finalCanvas, 0, 0);
        }
        
        // Step 3: Normalize Intensity (Conditional)
        if (options.useThinning) {
            const thinningValue = parseFloat(options.thinningIterations) || 0;
            const fullIterations = Math.floor(thinningValue);
            const partialIterationProb = thinningValue - fullIterations;

            const applyErosion = (probability = 1.0) => {
                const imageData = finalCtx.getImageData(0, 0, 28, 28);
                const data = imageData.data;
                const erodedData = new Uint8ClampedArray(data.length);

                for (let y = 0; y < 28; y++) {
                    for (let x = 0; x < 28; x++) {
                        const idx = (y * 28 + x) * 4;
                        
                        if (data[idx + 3] > 0) { // If pixel is not transparent
                            let isBoundary = false;
                            // Check 3x3 neighborhood
                            for (let ny = -1; ny <= 1; ny++) {
                                for (let nx = -1; nx <= 1; nx++) {
                                    if (nx === 0 && ny === 0) continue;
                                    const newX = x + nx;
                                    const newY = y + ny;
                                    
                                    // If a neighbor is off-canvas or is transparent, this pixel is on a boundary
                                    if (newX < 0 || newX >= 28 || newY < 0 || newY >= 28 || data[((newY * 28 + newX) * 4) + 3] === 0) {
                                        isBoundary = true;
                                        break;
                                    }
                                }
                                if (isBoundary) break;
                            }
                            
                            // If it's not a boundary pixel, it's an interior pixel. Keep it.
                            if (!isBoundary) {
                                erodedData[idx] = data[idx];
                                erodedData[idx + 1] = data[idx + 1];
                                erodedData[idx + 2] = data[idx + 2];
                                erodedData[idx + 3] = data[idx + 3];
                            } else {
                                // If it IS a boundary pixel, decide whether to erode it based on probability
                                if (Math.random() >= probability) {
                                    // Keep the pixel (do not erode)
                                    erodedData[idx] = data[idx];
                                    erodedData[idx + 1] = data[idx + 1];
                                    erodedData[idx + 2] = data[idx + 2];
                                    erodedData[idx + 3] = data[idx + 3];
                                }
                                // Otherwise, do nothing, effectively eroding it (leaving it transparent)
                            }
                        }
                    }
                }
                imageData.data.set(erodedData);
                finalCtx.putImageData(imageData, 0, 0);
            };

            // Apply full iterations
            for (let i = 0; i < fullIterations; i++) {
                applyErosion(1.0); // 100% probability
            }

            // Apply partial iteration
            if (partialIterationProb > 0) {
                applyErosion(partialIterationProb);
            }
        }

        return finalCanvas;
    }

    // --- Floating Preprocess Popover (smart positioning) ---
    function positionPreprocessPopover() {
        if (!preprocessSubMenu || !preprocessSubMenu.classList.contains('active')) return;

        const anchor = preprocessSettingsCog;
        if (!anchor) return;

        const rect = anchor.getBoundingClientRect();
        const margin = 10; // viewport margin
        const gap = 8;     // space between anchor and popover
        const desiredWidth = Math.min(320, window.innerWidth - margin * 2);

        // Main settings cog position (bottom of screen reference point)
        const settingsCogRect = settingsCog.getBoundingClientRect();
        const maxBottom = settingsCogRect.top - gap; // Just above the main cog

        // Prepare for measurement
        preprocessSubMenu.style.width = desiredWidth + 'px';
        preprocessSubMenu.style.maxHeight = 'none'; // Remove max-height temporarily to get natural height
        preprocessSubMenu.style.visibility = 'hidden';
        preprocessSubMenu.style.display = 'block';

        const popHeight = preprocessSubMenu.scrollHeight; // Natural height
        const availableBelow = maxBottom - rect.top;
        const availableAbove = rect.bottom - margin;

        let top, maxHeight;
        
        // Check if we can fit below the anchor
        if (availableBelow >= popHeight) {
            // Open downward - enough space
            top = rect.top;
            maxHeight = availableBelow;
        } else if (availableAbove >= popHeight) {
            // Open upward - enough space above
            top = rect.bottom - popHeight;
            maxHeight = availableAbove;
        } else {
            // Not enough space in either direction, choose the larger space
            if (availableBelow >= availableAbove) {
                // Open downward with scroll
                top = rect.top;
                maxHeight = availableBelow;
            } else {
                // Open upward with scroll
                top = margin;
                maxHeight = availableAbove;
            }
        }

        // Horizontal placement
        const availableRight = window.innerWidth - rect.right;
        const availableLeft = rect.left;

        let left;
        if (availableRight >= desiredWidth + gap) {
            // open to the right
            left = rect.right + gap;
        } else if (availableLeft >= desiredWidth + gap) {
            // open to the left
            left = rect.left - desiredWidth - gap;
        } else {
            // center-ish fallback within viewport
            left = Math.max(margin, Math.min(window.innerWidth - margin - desiredWidth, rect.left));
        }

        // Apply final position
        preprocessSubMenu.style.left = Math.round(left) + 'px';
        preprocessSubMenu.style.top = Math.round(top) + 'px';
        preprocessSubMenu.style.maxHeight = Math.round(maxHeight) + 'px';
        preprocessSubMenu.style.visibility = '';
    }

    function closeAllSubmenus() {
        closePreprocessPopover();
        closeQknnPopover();
        closeKnnPopover();
    }

    function openPreprocessPopover() {
        closeQknnPopover();
        closeKnnPopover();
        if (!preprocessSubMenu.classList.contains('active')) {
            preprocessSubMenu.classList.add('active');
        }
        positionPreprocessPopover();
    }

    function closePreprocessPopover() {
        preprocessSubMenu.classList.remove('active');
        // optional: clear inline position to avoid stale placement
        // preprocessSubMenu.style.left = '';
        // preprocessSubMenu.style.top = '';
        // preprocessSubMenu.style.display = '';
    }

    // --- Algorithm Configuration Popovers ---
    function positionAlgorithmPopover(popover, anchor) {
        if (!popover || !popover.classList.contains('active')) return;
        if (!anchor) return;

        const rect = anchor.getBoundingClientRect();
        const margin = 10;
        const gap = 8;
        const desiredWidth = Math.min(320, window.innerWidth - margin * 2);

        // Main settings cog position (bottom of screen reference point)
        const settingsCogRect = settingsCog.getBoundingClientRect();
        const maxBottom = settingsCogRect.top - gap; // Just above the main cog

        // Prepare for measurement
        popover.style.width = desiredWidth + 'px';
        popover.style.maxHeight = 'none'; // Remove max-height temporarily to get natural height
        popover.style.visibility = 'hidden';
        popover.style.display = 'block';

        const popHeight = popover.scrollHeight; // Natural height
        const availableBelow = maxBottom - rect.top;
        const availableAbove = rect.bottom - margin;

        let top, maxHeight;
        
        // Check if we can fit below the anchor
        if (availableBelow >= popHeight) {
            // Open downward - enough space
            top = rect.top;
            maxHeight = availableBelow;
        } else if (availableAbove >= popHeight) {
            // Open upward - enough space above
            top = rect.bottom - popHeight;
            maxHeight = availableAbove;
        } else {
            // Not enough space in either direction, choose the larger space
            if (availableBelow >= availableAbove) {
                // Open downward with scroll
                top = rect.top;
                maxHeight = availableBelow;
            } else {
                // Open upward with scroll
                top = margin;
                maxHeight = availableAbove;
            }
        }

        // Horizontal placement
        const availableRight = window.innerWidth - rect.right;
        const availableLeft = rect.left;

        let left;
        if (availableRight >= desiredWidth + gap) {
            left = rect.right + gap;
        } else if (availableLeft >= desiredWidth + gap) {
            left = rect.left - desiredWidth - gap;
        } else {
            left = Math.max(margin, rect.left - desiredWidth - gap);
        }

        // Apply final position
        popover.style.left = Math.round(left) + 'px';
        popover.style.top = Math.round(top) + 'px';
        popover.style.maxHeight = Math.round(maxHeight) + 'px';
        popover.style.visibility = '';
    }    function openQknnPopover() {
        // Don't open if disabled (PythonAnywhere mode)
        if (qknnSettingsCog.disabled) return;
        
        closePreprocessPopover();
        closeKnnPopover();
        if (!qknnSubMenu.classList.contains('active')) {
            qknnSubMenu.classList.add('active');
        }
        positionAlgorithmPopover(qknnSubMenu, qknnSettingsCog);
    }

    function closeQknnPopover() {
        qknnSubMenu.classList.remove('active');
    }

    function openKnnPopover() {
        // Don't open if disabled (PythonAnywhere mode)
        if (knnSettingsCog.disabled) return;
        
        closePreprocessPopover();
        closeQknnPopover();
        if (!knnSubMenu.classList.contains('active')) {
            knnSubMenu.classList.add('active');
        }
        positionAlgorithmPopover(knnSubMenu, knnSettingsCog);
    }

    function closeKnnPopover() {
        knnSubMenu.classList.remove('active');
    }

    // Update algorithm configs from sliders
    function updateQknnConfig() {
        const trainingSize = parseInt(qknnTrainingSize.value);
        const pcaPower = parseInt(qknnPcaComponents.value);
        const pcaComponents = Math.pow(2, pcaPower);
        const k = parseInt(qknnK.value);
        const shotsPower = parseInt(qknnShots.value);
        const shots = Math.pow(2, shotsPower);

        qknnTrainingSizeValue.textContent = trainingSize;
        qknnPcaComponentsValue.textContent = pcaComponents;
        qknnKValue.textContent = k;
        qknnShotsValue.textContent = shots;

        algorithmConfigs.qknn_sim.training_size = trainingSize;
        algorithmConfigs.qknn_sim.pca_components = pcaComponents;
        algorithmConfigs.qknn_sim.k = k;
        algorithmConfigs.qknn_sim.shots = shots;

        // Sync K value to kNN
        if (knnK.value !== k.toString()) {
            knnK.value = k;
            knnKValue.textContent = k;
            algorithmConfigs.knn_classical.k = k;
        }
    }

    function updateKnnConfig() {
        const trainingSize = parseInt(knnTrainingSize.value);
        const pcaComponents = parseInt(knnPcaComponents.value);
        const k = parseInt(knnK.value);

        knnTrainingSizeValue.textContent = trainingSize;
        knnPcaComponentsValue.textContent = pcaComponents;
        knnKValue.textContent = k;

        algorithmConfigs.knn_classical.training_size = trainingSize;
        algorithmConfigs.knn_classical.pca_components = pcaComponents;
        algorithmConfigs.knn_classical.k = k;

        // Sync K value to QkNN
        if (qknnK.value !== k.toString()) {
            qknnK.value = k;
            qknnKValue.textContent = k;
            algorithmConfigs.qknn_sim.k = k;
        }
    }

    // Training size editable value handlers
    function setupEditableTrainingSize(valueElement, sliderElement, configUpdateFn) {
        // Enable editing on hover
        valueElement.addEventListener('mouseenter', () => {
            valueElement.setAttribute('contenteditable', 'true');
        });

        valueElement.addEventListener('mouseleave', () => {
            if (document.activeElement !== valueElement) {
                valueElement.setAttribute('contenteditable', 'false');
            }
        });

        // Handle blur (user clicks away or tabs out)
        valueElement.addEventListener('blur', () => {
            valueElement.setAttribute('contenteditable', 'false');
            let newValue = parseInt(valueElement.textContent);
            
            // Validate and constrain
            if (isNaN(newValue) || newValue < 250) newValue = 250;
            if (newValue > 60000) newValue = 60000;
            
            // Round to nearest 250
            newValue = Math.round(newValue / 250) * 250;
            
            // Update slider and display
            sliderElement.value = newValue;
            valueElement.textContent = newValue;
            configUpdateFn();
        });

        // Handle Enter key to confirm
        valueElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                valueElement.blur();
            }
            // Only allow numbers
            if (e.key.length === 1 && !/[0-9]/.test(e.key)) {
                e.preventDefault();
            }
        });
    }



    // --- Debug View Functions ---
    let lastProcessedCanvas = null;

    async function updateDebugView() {
        if (!debugViewToggle.checked) return;
        
        debugArea.style.display = 'block';
        
        // Update user's processed drawing with dark background
        let processedCanvas;
        if (preprocessToggle.checked) {
            const options = {
                useCoM: comToggle.checked,
                useBlur: blurToggle.checked,
                blurAmount: blurSlider.value,
                useThinning: thinningToggle.checked,
                thinningIterations: thinningSlider.value
            };
            processedCanvas = preprocessCanvas(options);
        } else {
            processedCanvas = document.createElement('canvas');
            processedCanvas.width = 28;
            processedCanvas.height = 28;
            processedCanvas.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
        }
        
        // Store for later use
        lastProcessedCanvas = processedCanvas;
        
        // Draw with black background to match MNIST
        const userCtx = debugCanvasUser.getContext('2d');
        userCtx.fillStyle = 'black';
        userCtx.fillRect(0, 0, 140, 140);
        userCtx.drawImage(processedCanvas, 0, 0, 140, 140);
        
        // Fetch and display MNIST sample
        await fetchMnistSample();
    }

    async function fetchMnistSample() {
        const filterDigit = mnistDigitFilter.value;
        let url = currentBackendUrl.replace('/classify', '/get_random_mnist_image');
        
        // If user specified a digit filter, add it as query parameter
        if (filterDigit !== '' && filterDigit >= 0 && filterDigit <= 9) {
            url += `?digit=${filterDigit}`;
        }
        
        try {
            const response = await fetch(url);
            const data = await response.json();
            
            // Update label
            mnistLabel.textContent = data.label;
            
            // Draw MNIST image with black background
            const mnistCtx = debugCanvasMnist.getContext('2d');
            mnistCtx.fillStyle = 'black';
            mnistCtx.fillRect(0, 0, 140, 140);
            
            const imageData = mnistCtx.createImageData(28, 28);
            for (let i = 0; i < data.pixels.length; i++) {
                const p = data.pixels[i];
                imageData.data[i * 4] = p;     // R
                imageData.data[i * 4 + 1] = p; // G
                imageData.data[i * 4 + 2] = p; // B
                imageData.data[i * 4 + 3] = 255; // A
            }
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);
            
            mnistCtx.drawImage(tempCanvas, 0, 0, 140, 140);
        } catch (error) {
            console.error('Error fetching MNIST sample:', error);
            mnistLabel.textContent = 'Error';
        }
    }

    // --- API Call ---

    function getSelectedAlgorithms() {
        const checkedBoxes = document.querySelectorAll('input[name="algorithm"]:checked');
        // Return an array of the values, e.g., ["qknn_sim", "knn_classical"]
        // If none are checked, default to an empty array.
        return Array.from(checkedBoxes).map(cb => cb.value);
    }

    async function classifyDrawing() {
        let processedCanvas;

        if (preprocessToggle.checked) {
            const options = {
                useCoM: comToggle.checked,
                useBlur: blurToggle.checked,
                blurAmount: blurSlider.value,
                useThinning: thinningToggle.checked,
                thinningIterations: thinningSlider.value
            };
            processedCanvas = preprocessCanvas(options);
        } else {
            processedCanvas = document.createElement('canvas');
            processedCanvas.width = 28;
            processedCanvas.height = 28;
            processedCanvas.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
        }

        const tempCtx = processedCanvas.getContext('2d', { willReadFrequently: true });
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = [];
        for (let i = 3; i < imageData.data.length; i += 4) {
            pixels.push(imageData.data[i]);
        }

    loader.style.display = 'inline-block';
        classifyBtn.disabled = true;
        resetResultDisplay();
        lastResultState = null;
        
        // Update debug view if enabled
        if (debugViewToggle.checked) {
            await updateDebugView();
        }

        const selectedAlgoNames = getSelectedAlgorithms();
        if (selectedAlgoNames.length === 0) {
            loader.style.display = 'none';
            lastResultState = { type: 'message', message: 'Please select at least one algorithm.' };
            renderResultMessage('Please select at least one algorithm.');
            updateClassifyButtonState();
            return;
        }

        const apiPayload = selectedAlgoNames.map(name => ({
            name,
            config: algorithmConfigs[name]
        }));

        const includeTiming = showTimeToggle ? showTimeToggle.checked : true;

        try {
            const response = await fetch(currentBackendUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels, algorithms: apiPayload }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            const predictions = result.predictions || {};
            const timings = result.timings || {};

            const rows = selectedAlgoNames.map(name => ({
                key: name,
                displayName: getDisplayNameForAlgorithm(name),
                prediction: Object.prototype.hasOwnProperty.call(predictions, name) ? predictions[name] : 'N/A',
                time: timings[name]
            }));

            if (rows.length === 1) {
                lastResultState = { type: 'single', rows };
                renderSingleResult(rows[0], includeTiming);
            } else if (rows.length > 1) {
                lastResultState = { type: 'multi', rows };
                renderMultiResult(rows, includeTiming);
            } else {
                lastResultState = { type: 'message', message: 'No predictions returned.' };
                renderResultMessage('No predictions returned.');
            }

            if (historyToggle?.checked) {
                const configsForLog = {};
                apiPayload.forEach(algo => {
                    configsForLog[algo.name] = algo.config;
                });

                const primaryTime = rows[0]?.time;
                const newEntry = {
                    id: Date.now(),
                    imageDataUrl: processedCanvas.toDataURL(),
                    duration: typeof primaryTime === 'number' && primaryTime >= 0 ? primaryTime : 0,
                    correctLabel: null,
                    predictions,
                    configurations: configsForLog,
                    timings
                };

                historyLog.push(newEntry);
                saveHistory();
                renderHistoryTable();
            }

        } catch (error) {
            console.error('Error:', error);
            lastResultState = { type: 'message', message: 'Error during classification.' };
            renderResultMessage('Error during classification.');
        } finally {
            loader.style.display = 'none';
            updateClassifyButtonState();
        }
    }

    // --- END: NEW CODE BLOCK ---
    
    // --- 4. Initialization and Event Listeners ---
    
    function init() {
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) setTheme(savedTheme === 'dark');

        // Load saved settings
        loadHistory();
        loadSettings();
        applyBackendFromRadios();
        updateBackendStatusDots(); // Check status on page load

        // Dark Mode Toggle
        darkModeToggle.addEventListener('change', () => {
            const isDark = darkModeToggle.checked;
            setTheme(isDark);
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });

        // Hamburger Menu
        hamburger.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navbar.classList.toggle('open');
        });
        
        // Settings Menu
        if (settingsCog && settingsMenu) {
            settingsCog.addEventListener('click', () => {
                settingsMenu.classList.toggle('active');
                if (settingsMenu.classList.contains('active')) {
                    updateBackendStatusDots();
                }
            });
        }

        // Pencil Menu
        if (pencilCog && pencilMenu) {
            pencilCog.addEventListener('click', () => {
                pencilMenu.classList.toggle('active');
            });
        }

        // Settings Panel Listeners
        backendRadios.forEach(radio => radio.addEventListener('change', () => {
            saveSettings();
            applyBackendFromRadios();
            updateClassifyButtonState();
        }));
        algorithmRadios.forEach(radio => radio.addEventListener('change', saveSettings));
        showTimeToggle.addEventListener('change', () => {
            saveSettings();
            replayLastResult();
        });

        historyToggle.addEventListener('change', () => {
            historyContainer.classList.toggle('active', historyToggle.checked);
            // If the user is enabling the history view, render the table.
            if (historyToggle.checked) {
                renderHistoryTable();
            }
        });

        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the entire history? This cannot be undone.')) {
                historyLog = []; // Empty the array
                saveHistory(); // Update localStorage
                renderHistoryTable(); // Redraw the empty table
            }
        });

        historyTableBody.addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-row-btn')) {
                const entryId = parseInt(e.target.dataset.id);
                // Filter the log to keep every entry EXCEPT the one with the matching ID
                historyLog = historyLog.filter(entry => entry.id !== entryId);
                saveHistory();
                renderHistoryTable();
            }
        });
        
        // Global Click & Key Listeners
        document.addEventListener('click', (e) => {
            if (!navbar.contains(e.target)) {
                navLinks.classList.remove('active');
                navbar.classList.remove('open');
            }
            if (settingsMenu && !settingsMenu.contains(e.target) && !settingsCog.contains(e.target)) {
                settingsMenu.classList.remove('active');
            }
            if (pencilMenu && !pencilMenu.contains(e.target) && !pencilCog.contains(e.target)) {
                pencilMenu.classList.remove('active');
            }
            // Close preprocess submenu if clicking outside of it and the small cog
            if (preprocessSubMenu.classList.contains('active')) {
                const clickedInsidePre = preprocessSubMenu.contains(e.target);
                const clickedAnchor = preprocessSettingsCog.contains(e.target);
                if (!clickedInsidePre && !clickedAnchor) {
                    closePreprocessPopover();
                }
            }
            // Close QkNN submenu if clicking outside
            if (qknnSubMenu.classList.contains('active')) {
                const clickedInsideQknn = qknnSubMenu.contains(e.target);
                const clickedAnchorQknn = qknnSettingsCog.contains(e.target);
                if (!clickedInsideQknn && !clickedAnchorQknn) {
                    closeQknnPopover();
                }
            }
            // Close kNN submenu if clicking outside
            if (knnSubMenu.classList.contains('active')) {
                const clickedInsideKnn = knnSubMenu.contains(e.target);
                const clickedAnchorKnn = knnSettingsCog.contains(e.target);
                if (!clickedInsideKnn && !clickedAnchorKnn) {
                    closeKnnPopover();
                }
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                navLinks.classList.remove('active');
                navbar.classList.remove('open');
                settingsMenu?.classList.remove('active');
                pencilMenu?.classList.remove('active');
                closeAllSubmenus();
            }
        });

        historyTableBody.addEventListener('change', (e) => {
            // Check if the event was triggered by a correct-label-input field
            if (e.target.classList.contains('correct-label-input')) {
                const entryId = parseInt(e.target.dataset.id);
                const newLabel = e.target.value;

                // Find the corresponding entry in our log
                const entryToUpdate = historyLog.find(entry => entry.id === entryId);
                
                if (entryToUpdate) {
                    entryToUpdate.correctLabel = newLabel;
                    saveHistory(); // Persist the change
                    updateAccuracyTally(); // Recalculate and display the new scores
                }
            }
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth > MOBILE_BREAKPOINT) {
                navLinks.classList.remove('active');
                navbar.classList.remove('open');
            }
            if (preprocessSubMenu.classList.contains('active')) {
                positionPreprocessPopover();
            }
            if (qknnSubMenu.classList.contains('active')) {
                positionAlgorithmPopover(qknnSubMenu, qknnSettingsCog);
            }
            if (knnSubMenu.classList.contains('active')) {
                positionAlgorithmPopover(knnSubMenu, knnSettingsCog);
            }
        });

        // Preprocess submenu: floating panel toggle (like settings-menu)
        preprocessSettingsCog.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (preprocessSubMenu.classList.contains('active')) {
                closePreprocessPopover();
            } else {
                openPreprocessPopover();
            }
        });

        // Prevent clicks inside the panel from closing it
        preprocessSubMenu.addEventListener('click', (e) => e.stopPropagation());

        // QkNN submenu
        qknnSettingsCog.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (qknnSubMenu.classList.contains('active')) {
                closeQknnPopover();
            } else {
                openQknnPopover();
            }
        });
        qknnSubMenu.addEventListener('click', (e) => e.stopPropagation());

        // kNN submenu
        knnSettingsCog.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (knnSubMenu.classList.contains('active')) {
                closeKnnPopover();
            } else {
                openKnnPopover();
            }
        });
        knnSubMenu.addEventListener('click', (e) => e.stopPropagation());

        // Algorithm config sliders
        qknnTrainingSize.addEventListener('input', updateQknnConfig);
        qknnPcaComponents.addEventListener('input', updateQknnConfig);
        qknnK.addEventListener('input', updateQknnConfig);
        qknnShots.addEventListener('input', updateQknnConfig);

        knnTrainingSize.addEventListener('input', updateKnnConfig);
        knnPcaComponents.addEventListener('input', updateKnnConfig);
        knnK.addEventListener('input', updateKnnConfig);

        // Initialize config displays
        updateQknnConfig();
        updateKnnConfig();

        // Setup editable training size values
        setupEditableTrainingSize(qknnTrainingSizeValue, qknnTrainingSize, updateQknnConfig);
        setupEditableTrainingSize(knnTrainingSizeValue, knnTrainingSize, updateKnnConfig);

        // Pencil configuration listeners
        strokeWidthSlider.addEventListener('input', () => {
            drawingConfig.strokeWidth = parseInt(strokeWidthSlider.value);
            strokeWidthValue.textContent = strokeWidthSlider.value;
        });

        fountainPenToggle.addEventListener('change', () => {
            drawingConfig.fountainPen = fountainPenToggle.checked;
        });

        blurSlider.addEventListener('input', () => {
            blurValue.textContent = blurSlider.value;
        });

        thinningSlider.addEventListener('input', () => {
            thinningValue.textContent = thinningSlider.value;
        });

        function updateSliderStates() {
            blurSlider.disabled = !blurToggle.checked;
            thinningSlider.disabled = !thinningToggle.checked;
        }

        blurToggle.addEventListener('change', updateSliderStates);
        thinningToggle.addEventListener('change', updateSliderStates);

        // Initial state
        updateSliderStates();

        // Canvas Drawing Events
        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', endPosition);
        canvas.addEventListener('mouseleave', endPosition);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('touchstart', startPosition, { passive: false });
        canvas.addEventListener('touchend', endPosition);
        canvas.addEventListener('touchcancel', endPosition);
        canvas.addEventListener('touchmove', draw, { passive: false });

        // Buttons
        classifyBtn.addEventListener('click', classifyDrawing);
        clearBtn.addEventListener('click', clearCanvas);

        // Debug view toggle
        debugViewToggle.addEventListener('change', () => {
            if (debugViewToggle.checked) {
                debugArea.style.display = 'block';
                updateDebugView();
            } else {
                debugArea.style.display = 'none';
            }
        });

        // Compare with MNIST button
        compareMnistBtn.addEventListener('click', fetchMnistSample);

    }
    
    init();

});