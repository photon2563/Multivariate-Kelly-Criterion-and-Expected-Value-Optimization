const DOM = {
    tickerInput: document.getElementById('tickerInput'),
    fetchTickerBtn: document.getElementById('fetchTickerBtn'),
    spotPriceDisplay: document.getElementById('spotPriceDisplay'),
    expiryGroup: document.getElementById('expiryGroup'),
    expirySelect: document.getElementById('expirySelect'),
    ttmCard: document.getElementById('ttmCard'),
    ttmDisplay: document.getElementById('ttmDisplay'),
    apiStatus: document.getElementById('apiStatus'),
    loader: document.getElementById('loader'),
    loaderText: document.getElementById('loaderText'),
    calibrationResults: document.getElementById('calibrationResults'),
    chartContainer: document.getElementById('chartContainer'),
    valAlpha: document.getElementById('valAlpha'),
    valBeta: document.getElementById('valBeta'),
    valRho: document.getElementById('valRho'),
    valNu: document.getElementById('valNu'),
    valShift: document.getElementById('valShift'),
    valMse: document.getElementById('valMse')
};

let currentSpotPrice = 0;
let currentTimeToMaturity = 0;

function showLoader(text) {
    DOM.loaderText.innerText = text;
    DOM.loader.style.display = 'flex';
    DOM.calibrationResults.style.display = 'none';
    DOM.chartContainer.style.display = 'none';
}

function hideLoader() {
    DOM.loader.style.display = 'none';
}

DOM.fetchTickerBtn.addEventListener('click', async () => {
    const symbol = DOM.tickerInput.value.toUpperCase();
    if (!symbol) return;

    showLoader(`Fetching Options Chain for ${symbol}...`);
    DOM.apiStatus.innerText = "● Connecting to yfinance...";
    DOM.apiStatus.style.color = "#d2a8ff";

    try {
        const response = await fetch(`/api/v1/market_data/expirations/${symbol}`);
        if (!response.ok) throw new Error("Failed to fetch expirations.");
        const data = await response.json();
        
        currentSpotPrice = data.spot_price;
        DOM.spotPriceDisplay.innerText = `$${currentSpotPrice.toFixed(2)}`;
        
        // Populate expirations
        DOM.expirySelect.innerHTML = '';
        let defaultIndex = 0;
        const today = new Date();
        
        data.expirations.forEach((exp, index) => {
            const opt = document.createElement('option');
            opt.value = exp;
            opt.innerText = exp;
            DOM.expirySelect.appendChild(opt);
            
            const expDate = new Date(exp);
            const daysDiff = (expDate - today) / (1000 * 60 * 60 * 24);
            if (daysDiff > 30 && defaultIndex === 0) {
                defaultIndex = index;
            }
        });
        
        DOM.expirySelect.selectedIndex = defaultIndex;
        DOM.expiryGroup.style.display = 'flex';
        
        await handleExpiryChange();
        
    } catch (err) {
        alert(err.message);
        hideLoader();
        DOM.apiStatus.innerText = "● Error";
        DOM.apiStatus.style.color = "#f85149";
    }
});

DOM.expirySelect.addEventListener('change', handleExpiryChange);

async function handleExpiryChange() {
    const symbol = DOM.tickerInput.value.toUpperCase();
    const expiry = DOM.expirySelect.value;
    
    const expDate = new Date(expiry);
    const today = new Date();
    const daysDiff = (expDate - today) / (1000 * 60 * 60 * 24);
    currentTimeToMaturity = Math.max(daysDiff / 365.25, 0.001);
    
    DOM.ttmDisplay.innerText = `${currentTimeToMaturity.toFixed(4)} yrs`;
    DOM.ttmCard.style.display = 'flex';
    
    showLoader("Extracting Liquidity Profiles...");
    
    try {
        const response = await fetch(`/api/v1/market_data/options/${symbol}/${expiry}`);
        const data = await response.json();
        
        // Filter calls
        let calls = data.calls.filter(c => c.impliedVolatility > 0.01 && c.volume > 0);
        const lowerBound = currentSpotPrice * 0.8;
        const upperBound = currentSpotPrice * 1.2;
        calls = calls.filter(c => c.strike >= lowerBound && c.strike <= upperBound);
        
        if (calls.length < 5) {
            throw new Error("Not enough liquid options data for stable calibration.");
        }
        
        const strikes = calls.map(c => c.strike);
        const marketVols = calls.map(c => c.impliedVolatility);
        
        await runCalibration(strikes, marketVols);
        
    } catch (err) {
        alert(err.message);
        hideLoader();
    }
}

async function runCalibration(strikes, marketVols) {
    showLoader("Running SABR JIT Optimization...");
    DOM.apiStatus.innerText = "● Calibrating...";
    DOM.apiStatus.style.color = "#58a6ff";
    
    try {
        const payload = {
            forward_rate: currentSpotPrice, // Using spot as proxy for forward
            time_to_maturity: currentTimeToMaturity,
            strikes: strikes,
            market_vols: marketVols,
            beta: 1.0,
            shift: 0.0
        };
        
        const response = await fetch(`/api/v1/calibrate_sabr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("Calibration failed.");
        const results = await response.json();
        
        DOM.valAlpha.innerText = results.alpha.toFixed(4);
        DOM.valBeta.innerText = results.beta.toFixed(4);
        DOM.valRho.innerText = results.rho.toFixed(4);
        DOM.valNu.innerText = results.nu.toFixed(4);
        DOM.valShift.innerText = results.shift.toFixed(4);
        DOM.valMse.innerText = results.mse.toExponential(4);
        
        hideLoader();
        DOM.calibrationResults.style.display = 'grid';
        DOM.chartContainer.style.display = 'flex';
        DOM.apiStatus.innerText = "● Optimization Complete";
        DOM.apiStatus.style.color = "#238636";
        
        renderChart(strikes, marketVols, results);
        
    } catch (err) {
        alert(err.message);
        hideLoader();
    }
}

function sabrVol(F, K, T, alpha, beta, rho, nu, shift) {
    let f = F + shift;
    let k = K + shift;
    if (k <= 0) return 0;
    
    let logFk = Math.log(f / k);
    let fkBeta = Math.pow(f * k, (1 - beta) / 2);
    
    let term1 = fkBeta * (1 + Math.pow(1 - beta, 2) / 24 * Math.pow(logFk, 2) + Math.pow(1 - beta, 4) / 1920 * Math.pow(logFk, 4));
    
    let z = (nu / alpha) * Math.pow(f * k, (1 - beta) / 2) * logFk;
    
    let x;
    if (Math.abs(z) < 1e-7) {
        x = 1.0;
    } else {
        x = Math.log((Math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho));
    }
    
    let num = alpha;
    let den = term1;
    
    let factor1 = 1.0;
    if (Math.abs(z) >= 1e-7) {
        factor1 = z / x;
    }
    
    let factor2 = 1 + (
        Math.pow(1 - beta, 2) / 24 * (alpha * alpha) / (fkBeta * fkBeta) +
        (rho * beta * nu * alpha) / (4 * fkBeta) +
        (2 - 3 * rho * rho) / 24 * (nu * nu)
    ) * T;
    
    return (num / den) * factor1 * factor2;
}

function renderChart(strikes, marketVols, params) {
    const minStrike = Math.min(...strikes) * 0.9;
    const maxStrike = Math.max(...strikes) * 1.1;
    
    const denseStrikes = [];
    const sabrVols = [];
    
    for (let k = minStrike; k <= maxStrike; k += (maxStrike - minStrike)/100) {
        denseStrikes.push(k);
        const v = sabrVol(currentSpotPrice, k, currentTimeToMaturity, params.alpha, params.beta, params.rho, params.nu, params.shift);
        sabrVols.push(v);
    }
    
    const traceMarket = {
        x: strikes,
        y: marketVols,
        mode: 'markers',
        name: 'Market Implied Volatility',
        marker: { size: 8, color: '#58a6ff', symbol: 'diamond' }
    };
    
    const traceSABR = {
        x: denseStrikes,
        y: sabrVols,
        mode: 'lines',
        name: 'SABR Calibrated Surface',
        line: { color: '#238636', width: 3, shape: 'spline' }
    };
    
    const layout = {
        title: { text: 'Volatility Smile Calibration', font: { color: '#c9d1d9', size: 18 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { title: 'Strike Price', gridcolor: 'rgba(255,255,255,0.1)', tickfont: { color: '#8b949e' }, titlefont: { color: '#8b949e' } },
        yaxis: { title: 'Implied Volatility', gridcolor: 'rgba(255,255,255,0.1)', tickfont: { color: '#8b949e' }, titlefont: { color: '#8b949e' } },
        legend: { font: { color: '#c9d1d9' } },
        margin: { t: 50, l: 50, r: 20, b: 50 }
    };
    
    Plotly.newPlot('volatilityChart', [traceMarket, traceSABR], layout, {responsive: true});
}
