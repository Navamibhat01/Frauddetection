document.addEventListener('DOMContentLoaded', () => {
    // 43 features to simulate
    const featureList = [
        // Device Features (11)
        { name: "device.user_agent", type: "device" },
        { name: "device.screen_width", type: "device" },
        { name: "device.screen_height", type: "device" },
        { name: "device.touch_support", type: "device" },
        { name: "device.battery_level", type: "device" },
        { name: "device.battery_charging", type: "device" },
        { name: "device.os_version", type: "device" },
        { name: "device.hardware_concurrency", type: "device" },
        { name: "device.device_memory_gb", type: "device" },
        { name: "device.canvas_fingerprint", type: "device" },
        { name: "device.audio_fingerprint", type: "device" },
        // Temporal Features (7)
        { name: "temporal.local_hour", type: "temporal" },
        { name: "temporal.day_of_week", type: "temporal" },
        { name: "temporal.timezone_offset", type: "temporal" },
        { name: "temporal.session_duration_s", type: "temporal" },
        { name: "temporal.typing_speed_wpm", type: "temporal" },
        { name: "temporal.keystroke_jitter_ms", type: "temporal" },
        { name: "temporal.screen_idle_time_s", type: "temporal" },
        // Geospatial Features (9)
        { name: "geospatial.latitude", type: "geospatial" },
        { name: "geospatial.longitude", type: "geospatial" },
        { name: "geospatial.ip_address", type: "geospatial" },
        { name: "geospatial.vpn_detected", type: "geospatial" },
        { name: "geospatial.proxy_detected", type: "geospatial" },
        { name: "geospatial.tor_exit_node", type: "geospatial" },
        { name: "geospatial.network_provider", type: "geospatial" },
        { name: "geospatial.network_type", type: "geospatial" },
        { name: "geospatial.distance_from_home_km", type: "geospatial" },
        // Behavioral Features (7)
        { name: "behavioral.touch_velocity", type: "behavioral" },
        { name: "behavioral.scroll_cadence", type: "behavioral" },
        { name: "behavioral.accel_x", type: "behavioral" },
        { name: "behavioral.accel_y", type: "behavioral" },
        { name: "behavioral.accel_z", type: "behavioral" },
        { name: "behavioral.screen_share_active", type: "behavioral" },
        { name: "behavioral.root_status", type: "behavioral" },
        // Relational Features (9)
        { name: "relational.user_id", type: "relational" },
        { name: "relational.merchant_id", type: "relational" },
        { name: "relational.amount", type: "relational" },
        { name: "relational.payment_method", type: "relational" },
        { name: "relational.node_degree_user", type: "relational" },
        { name: "relational.node_degree_merchant", type: "relational" },
        { name: "relational.edge_density", type: "relational" },
        { name: "relational.user_fraud_ratio", type: "relational" },
        { name: "relational.merchant_risk_score", type: "relational" }
    ];

    // UI elements
    const upiForm = document.getElementById('upi-payment-form');
    const payBtn = document.getElementById('pay-btn');
    const resetBtn = document.getElementById('reset-btn');
    const walletScreen = document.getElementById('wallet-screen');
    const processingScreen = document.getElementById('processing-screen');
    const verdictScreen = document.getElementById('verdict-screen');
    
    const mobileLogStream = document.getElementById('mobile-log-stream');
    const mainLogStream = document.getElementById('main-log-stream');
    const processStepText = document.querySelector('.process-step-text');
    const featuresCounter = document.getElementById('features-counter');
    
    // Cards
    const mappingCard = document.getElementById('mapping-card');
    const xaiCard = document.getElementById('xai-card');
    const aggregatorStatus = document.getElementById('aggregator-status');
    const distanceOverlay = document.getElementById('distance-overlay');

    // Verdict elements
    const verdictIconBg = document.getElementById('verdict-icon-bg');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const verdictTxnId = document.getElementById('verdict-txn-id');
    const verdictRiskPct = document.getElementById('verdict-risk-pct');

    // XAI UI items
    const xaiOutlierBadge = document.getElementById('xai-outlier-badge');
    const xaiOutlierText = document.getElementById('xai-outlier-text');
    const xaiEdgeBadge = document.getElementById('xai-edge-badge');
    const xaiEdgeText = document.getElementById('xai-edge-text');
    const xaiVarianceBadge = document.getElementById('xai-variance-badge');
    const xaiVarianceText = document.getElementById('xai-variance-text');

    // Quick amount select buttons
    document.querySelectorAll('.btn-quick').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.getElementById('txn-amount').value = e.target.getAttribute('data-amount');
        });
    });

    // Canvas elements & contexts
    const netCanvas = document.getElementById('network-canvas');
    const latentCanvas = document.getElementById('latent-space-canvas');
    const netCtx = netCanvas.getContext('2d');
    const latentCtx = latentCanvas.getContext('2d');

    let animationFrameId;
    let netAnimationActive = false;
    let latentAnimationActive = false;

    // Set Canvas Dimensions
    function resizeCanvases() {
        const netParent = netCanvas.parentElement;
        netCanvas.width = netParent.clientWidth;
        netCanvas.height = netParent.clientHeight;
        
        const latentParent = latentCanvas.parentElement;
        latentCanvas.width = latentParent.clientWidth;
        latentCanvas.height = latentParent.clientHeight;
    }
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    // Initial draw of Latent Space Cloud (Global Safe Cluster)
    let safeClusterPoints = [];
    function generateSafeCluster(width, height) {
        safeClusterPoints = [];
        const centerX = width / 2;
        const centerY = height / 2;
        const count = 120;
        
        for (let i = 0; i < count; i++) {
            // Box-Muller transform for gaussian distribution
            let u = 0, v = 0;
            while(u === 0) u = Math.random();
            while(v === 0) v = Math.random();
            let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
            
            let u2 = 0, v2 = 0;
            while(u2 === 0) u2 = Math.random();
            while(v2 === 0) v2 = Math.random();
            let num2 = Math.sqrt( -2.0 * Math.log( u2 ) ) * Math.sin( 2.0 * Math.PI * v2 );
            
            // Standard deviation of 30 pixels
            const rX = num * 32;
            const rY = num2 * 26;
            
            safeClusterPoints.push({
                x: centerX + rX,
                y: centerY + rY,
                size: Math.random() * 2 + 1.2,
                alpha: Math.random() * 0.4 + 0.3
            });
        }
    }
    generateSafeCluster(latentCanvas.width, latentCanvas.height);

    function drawStaticLatentSpace() {
        latentCtx.clearRect(0, 0, latentCanvas.width, latentCanvas.height);
        
        // Draw grid
        latentCtx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        latentCtx.lineWidth = 1;
        const gridSize = 30;
        for (let x = 0; x < latentCanvas.width; x += gridSize) {
            latentCtx.beginPath();
            latentCtx.moveTo(x, 0);
            latentCtx.lineTo(x, latentCanvas.height);
            latentCtx.stroke();
        }
        for (let y = 0; y < latentCanvas.height; y += gridSize) {
            latentCtx.beginPath();
            latentCtx.moveTo(0, y);
            latentCtx.lineTo(latentCanvas.width, y);
            latentCtx.stroke();
        }

        // Draw cluster points
        latentCtx.fillStyle = 'rgba(16, 185, 129, 0.6)';
        safeClusterPoints.forEach(p => {
            latentCtx.beginPath();
            latentCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            latentCtx.fill();
        });

        // Safe cluster core label
        latentCtx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        latentCtx.beginPath();
        latentCtx.arc(latentCanvas.width / 2, latentCanvas.height / 2, 45, 0, Math.PI * 2);
        latentCtx.fill();
    }
    drawStaticLatentSpace();

    // Reset workflow
    resetBtn.addEventListener('click', () => {
        verdictScreen.classList.remove('active');
        walletScreen.classList.add('active');
        payBtn.disabled = false;
        
        // Disable cards
        mappingCard.classList.add('disabled');
        xaiCard.classList.add('disabled');
        
        // Reset logs
        mobileLogStream.innerHTML = '';
        mainLogStream.innerHTML = '<div class="empty-terminal-msg">Waiting for transaction initiation...</div>';
        featuresCounter.textContent = '0 / 43 Features';
        featuresCounter.className = 'badge';
        aggregatorStatus.textContent = 'Idle';
        aggregatorStatus.className = 'card-badge';
        distanceOverlay.style.opacity = '0';
        
        // Reset canvases
        netAnimationActive = false;
        latentAnimationActive = false;
        netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);
        drawStaticLatentSpace();
    });

    // Handle Form Submit
    upiForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const upiId = document.getElementById('upi-id').value;
        const amount = parseFloat(document.getElementById('txn-amount').value);
        
        // Get threat simulation toggles
        const screenShare = document.getElementById('sim-screen-share').checked;
        const rooted = document.getElementById('sim-root-status').checked;
        const vpn = document.getElementById('sim-vpn').checked;
        
        // --- 1. THE INVISIBLE SNIFFER: FEATURE VECTORIZATION ---
        // Construct the 43-dimensional numerical feature vector (x)
        const x = Array(43).fill(0.02); // Initialize with baseline noise
        
        // Populate features based on context inputs
        x[15] = screenShare ? 0.35 : 0.75; // temporal.typing_speed_wpm (threat/screen share is slower/uneven)
        x[16] = screenShare ? 0.55 : 0.12; // temporal.keystroke_jitter_ms
        x[21] = vpn ? 1.0 : 0.0;            // geospatial.vpn_detected
        x[22] = vpn ? 1.0 : 0.0;            // geospatial.proxy_detected
        x[26] = vpn ? 8.24 : 0.01;          // geospatial.distance_from_home_km (VPN projects high distance)
        x[33] = screenShare ? 1.0 : 0.0;    // behavioral.screen_share_active
        x[34] = rooted ? 1.0 : 0.0;         // behavioral.root_status
        x[37] = amount / 100000;            // relational.amount (normalized)
        x[38] = upiId.includes("merchant") ? 1.0 : 0.0; // relational.merchant_id target
        x[42] = rooted ? 0.85 : 0.0;        // relational.user_fraud_ratio
        
        // --- 2. GNN LATENT ENCODER PROJECTION (W * x) ---
        // Pre-defined GNN projection weight matrix (4 dimensions x 43 features)
        const W = Array.from({ length: 4 }, () => Array(43).fill(0.0));
        
        // Row 0 maps screen sharing and rooted flags to dimension 0
        W[0][33] = 1.2; 
        W[0][34] = 1.5;
        // Row 1 maps VPN and geographical distance to dimension 1
        W[1][21] = 1.0; 
        W[1][26] = 0.8;
        // Row 2 maps amount scale and historical fraud ratio to dimension 2
        W[2][37] = 0.9; 
        W[2][42] = 0.6;
        // Row 3 maps behavioral speed anomalies to dimension 3
        W[3][15] = -0.5; 
        W[3][16] = 0.4;
        
        // Compute raw GNN embedding and apply activation function (tanh)
        const z = Array(4).fill(0);
        for (let i = 0; i < 4; i++) {
            let sum = 0;
            for (let j = 0; j < 43; j++) {
                sum += W[i][j] * x[j];
            }
            z[i] = Math.tanh(sum); // Projects to [-1, 1]
        }
        
        // --- 3. UNSUPERVISED GNN DISTANCE COMPUTATIONS ---
        // Centroid & Covariance diagonals of the Global Safe Cluster
        const mu = [0.1, 0.05, 0.05, -0.05];
        const sigma_var = [0.18, 0.18, 0.15, 0.18];
        
        // A. Calculate Mahalanobis Distance: sqrt( sum( (z_i - mu_i)^2 / var_i ) )
        let sum_dist = 0;
        for (let i = 0; i < 4; i++) {
            sum_dist += Math.pow(z[i] - mu[i], 2) / Math.pow(sigma_var[i], 2);
        }
        const mahalanobis_dist = Math.sqrt(sum_dist);
        
        // B. Calculate Cosine Similarity: (z . mu) / (||z|| * ||mu||)
        let dot = 0, normZ = 0, normMu = 0;
        for (let i = 0; i < 4; i++) {
            dot += z[i] * mu[i];
            normZ += z[i] * z[i];
            normMu += mu[i] * mu[i];
        }
        const cos_sim = dot / (Math.sqrt(normZ) * Math.sqrt(normMu) || 1);
        
        // --- 4. MODEL VERDICT TRIGGER (NO HARDCODED LIMITS) ---
        // Outlier condition: Mahalanobis Distance is > 3.0σ OR Cosine Similarity < 0.75
        const isAnomaly = (mahalanobis_dist > 3.0) || (cos_sim < 0.75);
        
        const distanceSigma = mahalanobis_dist.toFixed(1);
        const edgeProb = (1 / (1 + Math.exp(mahalanobis_dist * 2 - 4.5))).toFixed(3);
        const riskPercentage = Math.min(99.9, Math.max(1.2, (1 - cos_sim) * 120 + mahalanobis_dist * 10)).toFixed(1);
        
        // Calculate exact coordinates in Latent Space relative to Centroid
        const centerX = latentCanvas.width / 2;
        const centerY = latentCanvas.height / 2;
        const scaleX = latentCanvas.width * 0.45;
        const scaleY = latentCanvas.height * 0.45;
        
        const targetX = Math.max(15, Math.min(latentCanvas.width - 15, centerX + (z[0] - mu[0]) * scaleX));
        const targetY = Math.max(15, Math.min(latentCanvas.height - 15, centerY + (z[1] - mu[1]) * scaleY));

        startInferencePipeline(upiId, amount, screenShare, rooted, vpn, isAnomaly, distanceSigma, edgeProb, riskPercentage, targetX, targetY);
    });

    function startInferencePipeline(upiId, amount, screenShare, rooted, vpn, isAnomaly, distanceSigma, edgeProb, riskPercentage, targetX, targetY) {
        payBtn.disabled = true;
        walletScreen.classList.remove('active');
        processingScreen.classList.add('active');
        processStepText.textContent = "Extracting implicit features...";
        
        // Enable sniffer card
        mappingCard.classList.remove('disabled');
        
        // Clear log placeholder
        mainLogStream.innerHTML = '';

        // 1. PHASE 1: Feature Sniffing Animation (1.5 seconds)
        let featureIndex = 0;
        const totalFeatures = featureList.length;
        
        function sniffFeature() {
            if (featureIndex < totalFeatures) {
                const feat = featureList[featureIndex];
                let mockValue = generateMockFeatureValue(feat.name, {upiId, amount, screenShare, rooted, vpn});
                
                // Add to terminal streams
                const timestamp = new Date().toLocaleTimeString().split(' ')[0];
                const logLine = document.createElement('div');
                logLine.className = 'log-line';
                
                // Custom coloring for suspicious values
                let valClass = 'success-term';
                if ((feat.name.includes('screen_share') && screenShare) ||
                    (feat.name.includes('root_status') && rooted) ||
                    (feat.name.includes('vpn_detected') && vpn) ||
                    (feat.name.includes('amount') && amount > 50000)) {
                    valClass = 'error-term';
                }
                
                logLine.innerHTML = `<span class="timestamp">[${timestamp}]</span> <span class="dim-term">${feat.name}:</span> <span class="${valClass}">${mockValue}</span>`;
                
                // Append to both terminals
                mobileLogStream.appendChild(logLine.cloneNode(true));
                mainLogStream.appendChild(logLine);
                
                // Auto scroll
                mobileLogStream.scrollTop = mobileLogStream.scrollHeight;
                mainLogStream.scrollTop = mainLogStream.scrollHeight;
                
                featureIndex++;
                featuresCounter.textContent = `${featureIndex} / ${totalFeatures} Features`;
                
                if (featureIndex === 25) {
                    featuresCounter.className = "badge badge-accent";
                }
                
                // Dynamically pace the sniffer to finish in 1.5s
                setTimeout(sniffFeature, 1500 / totalFeatures);
            } else {
                // Done sniffing, move to GraphSAGE Graph Construction
                triggerGraphSAGEPhase();
            }
        }
        
        sniffFeature();

        // 2. PHASE 2: GraphSAGE Message Passing Animation
        function triggerGraphSAGEPhase() {
            processStepText.textContent = "Aggregating neighbor embeddings (GraphSAGE Style)...";
            aggregatorStatus.textContent = "Aggregating...";
            aggregatorStatus.className = "card-badge badge-accent";
            
            // Activate Network Canvas animation
            netAnimationActive = true;
            animateNetworkConstruction(isAnomaly);
            
            setTimeout(() => {
                // Phase 3: Calculate Edge Weights
                processStepText.textContent = "Calculating edge reconstruction weights...";
                
                setTimeout(() => {
                    // Phase 4: Latent Space Projection
                    triggerLatentSpaceProjection();
                }, 1500);
                
            }, 1500);
        }

        // 3. PHASE 3: Latent Space Projection
        function triggerLatentSpaceProjection() {
            processStepText.textContent = "Measuring Latent Distance to Safe Cluster...";
            
            // Enable XAI Card
            xaiCard.classList.remove('disabled');
            
            latentAnimationActive = true;
            animateLatentSpaceProjection(isAnomaly, distanceSigma, edgeProb, targetX, targetY);
            
            setTimeout(() => {
                // Complete Pipeline & show Verdict screen
                showVerdict(isAnomaly, amount, riskPercentage, distanceSigma);
            }, 2500);
        }
    }

    // Helper: Generates realistic sniffing logs on the fly
    function generateMockFeatureValue(name, ctx) {
        if (name === "device.user_agent") return navigator.userAgent.slice(0, 30) + "...";
        if (name === "device.screen_width") return window.screen.width + "px";
        if (name === "device.screen_height") return window.screen.height + "px";
        if (name === "device.touch_support") return ('ontouchstart' in window) ? "true" : "false";
        if (name === "device.battery_level") return "0.87";
        if (name === "device.battery_charging") return "false";
        if (name === "device.os_version") return navigator.userAgent.includes("Windows") ? "Windows NT 10.0" : "OSX 12.4";
        if (name === "device.hardware_concurrency") return navigator.hardwareConcurrency || 8;
        if (name === "device.device_memory_gb") return navigator.deviceMemory || 8;
        if (name === "device.canvas_fingerprint") return "sha256:7f08aa43";
        if (name === "device.audio_fingerprint") return "db:10.45781";
        
        if (name === "temporal.local_hour") return new Date().getHours();
        if (name === "temporal.day_of_week") return new Date().getDay();
        if (name === "temporal.timezone_offset") return new Date().getTimezoneOffset();
        if (name === "temporal.session_duration_s") return (45 + Math.random() * 120).toFixed(0);
        if (name === "temporal.typing_speed_wpm") return ctx.amount > 50000 ? "42" : "68";
        if (name === "temporal.keystroke_jitter_ms") return (8 + Math.random() * 45).toFixed(1);
        if (name === "temporal.screen_idle_time_s") return "0.45";
        
        if (name === "geospatial.latitude") return "12.9716";
        if (name === "geospatial.longitude") return "77.5946";
        if (name === "geospatial.ip_address") return ctx.vpn ? "45.132.227.18" : "103.241.12.89";
        if (name === "geospatial.vpn_detected") return ctx.vpn ? "true [MOCK_ISP_PROXY]" : "false";
        if (name === "geospatial.proxy_detected") return ctx.vpn ? "true" : "false";
        if (name === "geospatial.tor_exit_node") return "false";
        if (name === "geospatial.network_provider") return ctx.vpn ? "M247 Ltd" : "Reliance Jio";
        if (name === "geospatial.network_type") return "cellular";
        if (name === "geospatial.distance_from_home_km") return ctx.vpn ? "8244.5" : "1.2";
        
        if (name === "behavioral.touch_velocity") return "0.34 px/ms";
        if (name === "behavioral.scroll_cadence") return "0.0";
        if (name === "behavioral.accel_x") return (Math.random() * 0.2).toFixed(3);
        if (name === "behavioral.accel_y") return (9.8 + Math.random() * 0.1).toFixed(3);
        if (name === "behavioral.accel_z") return (Math.random() * 0.2).toFixed(3);
        if (name === "behavioral.screen_share_active") return ctx.screenShare ? "true [DISCORD_SDK]" : "false";
        if (name === "behavioral.root_status") return ctx.rooted ? "true [SU_BINARY_EXISTS]" : "false";
        
        if (name === "relational.user_id") return ctx.upiId.split('@')[0];
        if (name === "relational.merchant_id") return ctx.upiId.includes("merchant") ? ctx.upiId : "merchant_txn_node";
        if (name === "relational.amount") return "₹" + ctx.amount;
        if (name === "relational.payment_method") return "UPI_INTENT";
        if (name === "relational.node_degree_user") return "14";
        if (name === "relational.node_degree_merchant") return "418";
        if (name === "relational.edge_density") return "0.045";
        if (name === "relational.user_fraud_ratio") return ctx.rooted ? "0.85" : "0.00";
        if (name === "relational.merchant_risk_score") return "0.012";
        
        return "N/A";
    }

    // Graph Construction Animation
    function animateNetworkConstruction(isAnomaly) {
        let startTime = null;
        const nodes = [
            { id: 'txn', label: 'Txn (Target)', x: netCanvas.width / 2, y: netCanvas.height / 2, color: varColor('--color-primary'), size: 14 },
            { id: 'user', label: 'User Node', x: netCanvas.width * 0.25, y: netCanvas.height * 0.3, color: '#3b82f6', size: 10 },
            { id: 'device', label: 'Device Node', x: netCanvas.width * 0.75, y: netCanvas.height * 0.35, color: '#f59e0b', size: 10 },
            { id: 'merchant', label: 'Merchant Node', x: netCanvas.width * 0.5, y: netCanvas.height * 0.8, color: '#ec4899', size: 10 }
        ];

        const edges = [
            { from: 'user', to: 'txn' },
            { from: 'device', to: 'txn' },
            { from: 'merchant', to: 'txn' }
        ];

        function tick(timestamp) {
            if (!netAnimationActive) return;
            if (!startTime) startTime = timestamp;
            const progress = (timestamp - startTime) / 3000; // 3 seconds loop

            netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);

            // Draw link background
            netCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            netCtx.lineWidth = 2;
            edges.forEach(e => {
                const fromNode = nodes.find(n => n.id === e.from);
                const toNode = nodes.find(n => n.id === e.to);
                netCtx.beginPath();
                netCtx.moveTo(fromNode.x, fromNode.y);
                netCtx.lineTo(toNode.x, toNode.y);
                netCtx.stroke();
            });

            // Animate GraphSAGE Aggregation (pulses running down edges to Target)
            edges.forEach(e => {
                const fromNode = nodes.find(n => n.id === e.from);
                const toNode = nodes.find(n => n.id === e.to);
                
                // 3 pulses traveling
                for (let i = 0; i < 3; i++) {
                    const offset = (progress * 1.5 + i * 0.33) % 1.0;
                    const px = fromNode.x + (toNode.x - fromNode.x) * offset;
                    const py = fromNode.y + (toNode.y - fromNode.y) * offset;
                    
                    netCtx.fillStyle = isAnomaly ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)';
                    netCtx.beginPath();
                    netCtx.arc(px, py, 4, 0, Math.PI * 2);
                    netCtx.fill();
                }
            });

            // Draw Nodes
            nodes.forEach(n => {
                // Glow effect
                netCtx.shadowBlur = 12;
                netCtx.shadowColor = n.color;
                
                netCtx.fillStyle = n.color;
                netCtx.beginPath();
                netCtx.arc(n.x, n.y, n.size, 0, Math.PI * 2);
                netCtx.fill();
                
                // Clear shadow configuration
                netCtx.shadowBlur = 0;

                // Node Labels
                netCtx.fillStyle = '#94a3b8';
                netCtx.font = '10px Space Grotesk';
                netCtx.textAlign = 'center';
                netCtx.fillText(n.label, n.x, n.y - n.size - 6);
            });

            requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // Latent Space Projection Animation
    function animateLatentSpaceProjection(isAnomaly, distanceSigma, edgeProb, targetX, targetY) {
        let startTime = null;
        
        // Target coordinates in Latent Space
        const centerX = latentCanvas.width / 2;
        const centerY = latentCanvas.height / 2;

        // Current positions for animation
        let currentX = 0;
        let currentY = 0;

        function tick(timestamp) {
            if (!latentAnimationActive) return;
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, elapsed / 2000); // 2s projection animation

            // Easing function (easeOutCubic)
            const ease = 1 - Math.pow(1 - progress, 3);

            // Starting point is top-left
            currentX = 20 + (targetX - 20) * ease;
            currentY = 20 + (targetY - 20) * ease;

            drawStaticLatentSpace();

            // Draw line to cluster centroid
            latentCtx.beginPath();
            latentCtx.setLineDash([4, 4]);
            latentCtx.strokeStyle = isAnomaly ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)';
            latentCtx.moveTo(centerX, centerY);
            latentCtx.lineTo(currentX, currentY);
            latentCtx.stroke();
            latentCtx.setLineDash([]); // Reset line dash

            // Draw the moving projection node
            const glowColor = isAnomaly ? '239, 68, 68' : '16, 185, 129';
            latentCtx.shadowBlur = 15;
            latentCtx.shadowColor = `rgba(${glowColor}, 0.8)`;
            latentCtx.fillStyle = `rgb(${glowColor})`;
            
            // Add pulse sizing
            const pulseSize = 6 + Math.sin(timestamp / 150) * 1.5;
            latentCtx.beginPath();
            latentCtx.arc(currentX, currentY, pulseSize, 0, Math.PI * 2);
            latentCtx.fill();
            latentCtx.shadowBlur = 0; // reset shadow

            // Display distance value dynamically during progress
            const liveDist = (distanceSigma * ease).toFixed(1);
            distanceOverlay.querySelector('.dist-val').textContent = liveDist + ' σ';
            distanceOverlay.style.opacity = '1';

            if (progress < 1.0) {
                requestAnimationFrame(tick);
            } else {
                // Done Animating projection
                aggregatorStatus.textContent = isAnomaly ? "Outlier Blocked" : "Verified Safe";
                aggregatorStatus.className = isAnomaly ? "card-badge badge-accent red-badge" : "card-badge badge-accent green-badge";
                
                // Show XAI final values
                updateXAI(isAnomaly, distanceSigma, edgeProb);
            }
        }
        requestAnimationFrame(tick);
    }

    // Unsupervised XAI explanation text updates
    function updateXAI(isAnomaly, sigma, prob) {
        if (isAnomaly) {
            // Metric 1: Node Outlier Detected
            xaiOutlierBadge.textContent = "Outlier";
            xaiOutlierBadge.className = "metric-status danger";
            xaiOutlierText.innerHTML = `Node Outlier Detected: Current embedding is <span class="error-term">${sigma}σ</span> from the centroids of the latent cluster.`;
            
            // Metric 2: Edge Reconstruction Loss
            xaiEdgeBadge.textContent = "High Loss";
            xaiEdgeBadge.className = "metric-status danger";
            xaiEdgeText.innerHTML = `Edge Reconstruction Loss: The relational probability of this Device-User-Merchant triangle is <span class="error-term">${prob}</span>.`;

            // Metric 3: Latent Space Deviation
            xaiVarianceBadge.textContent = "High Variance";
            xaiVarianceBadge.className = "metric-status danger";
            xaiVarianceText.innerHTML = `Latent Space Deviation: <span class="error-term">High variance</span> detected in the temporal and geospatial feature dimensions.`;
        } else {
            // Metric 1: Normal
            xaiOutlierBadge.textContent = "Normal";
            xaiOutlierBadge.className = "metric-status success";
            xaiOutlierText.innerHTML = `Node In-bounds: Current embedding is <span class="success-term">${sigma}σ</span> from the centroids of the latent cluster.`;

            // Metric 2: Stable
            xaiEdgeBadge.textContent = "Low Loss";
            xaiEdgeBadge.className = "metric-status success";
            xaiEdgeText.innerHTML = `Edge Reconstruction Safe: The relational probability of this Device-User-Merchant triangle is <span class="success-term">${prob}</span>.`;

            // Metric 3: Homogeneous
            xaiVarianceBadge.textContent = "Stable";
            xaiVarianceBadge.className = "metric-status success";
            xaiVarianceText.innerHTML = `Latent Space Match: Stable variance detected across all feature dimensions.`;
        }
    }

    function showVerdict(isAnomaly, amount, riskPct, sigma) {
        processingScreen.classList.remove('active');
        verdictScreen.classList.add('active');

        // Formatted transaction ID
        const mockTxn = "TXN-" + Math.floor(10000000 + Math.random() * 90000000);
        verdictTxnId.textContent = mockTxn;
        verdictRiskPct.textContent = riskPct + "%";
        
        // Color classification based on GNN Risk
        if (isAnomaly) {
            verdictIconBg.className = "verdict-icon-container danger-mode";
            verdictIcon.textContent = "✕";
            verdictTitle.textContent = "Transaction Blocked";
            verdictTitle.style.color = "var(--color-threat)";
            verdictDesc.innerHTML = `Security Alert: Latent outlier detected at <span class="error-term">${sigma}σ</span> deviation limit.`;
            
            // Add red glowing borders to simulator screen
            document.querySelector('.phone-screen').style.borderColor = "var(--color-threat)";
        } else {
            verdictIconBg.className = "verdict-icon-container success-mode";
            verdictIcon.textContent = "✓";
            verdictTitle.textContent = "Payment Authorized";
            verdictTitle.style.color = "var(--color-safe)";
            verdictDesc.innerHTML = `Success: Embedding verified at <span class="success-term">${sigma}σ</span> cluster proximity.`;
            
            // Add green glowing borders to simulator screen
            document.querySelector('.phone-screen').style.borderColor = "var(--color-safe)";
        }
    }

    // Helper helper: parses CSS variables
    function varColor(name) {
        return getComputedStyle(document.body).getPropertyValue(name).trim();
    }
});
