// app_logic.js (FINAL, CLEANED VERSION)

const BACKEND_URL = "http://127.0.0.1:8000";

// --- CORE ASYNCHRONOUS FETCH FUNCTION ---
async function fetchApi(endpoint, method = 'GET', body = null) {
    const options = {
        method: method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
    }
    
    const response = await fetch(`${BACKEND_URL}${endpoint}`, options);
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
        return { ok: response.ok, status: response.status, data: await response.json() };
    }
    return { ok: response.ok, status: response.status, data: await response.blob() };
}

// 1. Prediction/Analysis Function (Called by 'Run Analysis' button)
async function runPrediction(ticker) {
    const resultContainer = document.getElementById('result-container');
    resultContainer.innerHTML = '<h2>Analyzing... ⏳</h2><p>Training Hybrid Model...</p>';

    const apiResult = await fetchApi(`/predict/${ticker}`);

    if (apiResult.ok) {
        const data = apiResult.data;
        resultContainer.innerHTML = `
            <h2>Results for ${data.ticker}</h2>
            <p class="text-sm text-gray-500 dark:text-gray-400">Last updated: ${new Date(data.timestamp).toLocaleTimeString()}</p>
            <hr class="my-3 border-gray-300"/>
            <div class="result-metrics">
                <p class="text-lg"><strong>Quantum Model Accuracy:</strong> 
                    <span class="text-2xl font-bold text-primary">${(data.quantum_accuracy * 100).toFixed(2)}%</span>
                </p>
                <p><strong>Classical Model Error (MSE):</strong> ${data.classical_mse.toFixed(6)}</p>
            </div>
        `;
    } else {
        const errorDetail = (typeof apiResult.data === 'object' && apiResult.data !== null && apiResult.data.detail) ? apiResult.data.detail : "Server failed to execute the model. Check Python terminal.";
        resultContainer.innerHTML = `<h2 style="color: red;">Analysis Failed! (${apiResult.status})</h2><p>Error: ${errorDetail}</p>`;
    }
}

// 2. Dashboard Data Fetch (loadDashboardMetrics function)
async function loadDashboardMetrics() {
    // 1. Fetch data from two endpoints
    const metricsPromise = fetchApi(`/analysis/dashboard`);
    const historyPromise = fetchApi(`/predictions/recent`); 

    const [metricsResult, historyResult] = await Promise.all([metricsPromise, historyPromise]);

    if (metricsResult.ok) {
        const data = metricsResult.data;
        
        // --- Update Key Insight Cards ---
        // Note: I'm making up dummy IDs here based on common dashboard structures
        document.getElementById('quantum-accuracy-card').textContent = `${(data.quantum_accuracy * 100).toFixed(1)}%`;
        document.getElementById('classical-mse-benchmark').textContent = data.classical_mse.toFixed(3);
        document.getElementById('pnl-value').textContent = data.pnl_percentage; 
        document.getElementById('risk-assessment-value').textContent = (data.risk_score > 70 ? 'High' : (data.risk_score > 40 ? 'Medium' : 'Low'));
        
        // --- Update Risk Bar Visualization ---
        const riskBar = document.getElementById('risk-low-bar'); 
        if (riskBar) {
            riskBar.style.width = `${data.risk_score}%`; 
        }
    }

    if (historyResult.ok) {
        console.log("Recent Predictions Data Loaded:", historyResult.data);
        // You would add code here to populate a table if you had one in main_app.html
    }
}

// 3. Report Generation Handler 
async function generateReportHandler(event) {
    event.preventDefault(); 

    const dateRange = document.getElementById('date-range').value;
    const format = document.querySelector('input[name="export-format"]:checked').value;
    const generateButton = document.getElementById('generate-report-button');
    
    generateButton.textContent = `Generating ${format.toUpperCase()}...`;
    
    const config = {
        date_start: "2024-01-01", 
        date_end: "2024-12-31",
        report_format: format,
        scope: dateRange
    };

    const apiResult = await fetchApi('/report/generate', 'POST', config);
    generateButton.textContent = `Generate & Export`;

    if (apiResult.ok) {
        // Success: Trigger file download
        const url = window.URL.createObjectURL(apiResult.data);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Quantum_Report_${format}.${format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        alert(`Report Downloaded: ${format.toUpperCase()}`);
    } else {
        // Attempt to parse JSON error message if possible
        const errorBlob = apiResult.data;
        const errorDetail = await (errorBlob.text ? errorBlob.text() : Promise.resolve("Server error"));
        alert(`Report Generation Failed: ${errorDetail}`);
    }
}

// -----------------------------------------------------------
// --- GLOBAL LISTENER BINDING FUNCTION (Called by main_app.html) ---
// -----------------------------------------------------------

// ✅ यह फ़ंक्शन आपके main_app.html में navigateTo फ़ंक्शन द्वारा कॉल किया जाएगा
// app_logic.js (Add this new function and ensure it's available globally)

// app_logic.js (bindAnalysisListeners फ़ंक्शन के अंदर)

function bindAnalysisListeners() {
    const analyzeButton = document.getElementById('analyze-button');
    const tickerInput = document.getElementById('ticker-input');
    const resultContainer = document.getElementById('result-container');

    // नया हैंडलर फ़ंक्शन
    function handleAnalysisTrigger() {
        const ticker = tickerInput.value.toUpperCase();
        if (ticker) {
            runPrediction(ticker); // Call the correct function
        } else {
            resultContainer.innerHTML = `<p class="text-sm text-red-500">Please enter a stock ticker.</p>`;
        }
    }
    
    // 1. Click Event Binding
    if (analyzeButton) {
        // पुरानी event listeners हटाएँ और नया जोड़ें
        analyzeButton.removeEventListener('click', handleAnalysisTrigger); 
        analyzeButton.addEventListener('click', handleAnalysisTrigger); 
    }

    // 2. Enter Key Binding
    if (tickerInput) {
        // पुरानी event listeners हटाएँ
        tickerInput.removeEventListener('keypress', handleEnterKey); 
        
        function handleEnterKey(e) {
            if (e.key === 'Enter') {
                e.preventDefault(); 
                handleAnalysisTrigger(); 
            }
        }
        tickerInput.addEventListener('keypress', handleEnterKey);
    }
    
    // ... (generateReportHandler लॉजिक यहाँ से जारी)
}