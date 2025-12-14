// Dissolving CAPTCHA Game
// The CAPTCHA that refuses to be read

const captchaText = document.getElementById('captchaText');
const captchaInput = document.getElementById('captchaInput');
const verifyBtn = document.getElementById('verifyBtn');
const refreshBtn = document.getElementById('refreshBtn');
const feedback = document.getElementById('feedback');
const progressFill = document.getElementById('progressFill');
const attemptCounter = document.getElementById('attemptCounter');
const difficultyBadge = document.getElementById('difficultyBadge');
const timerWarning = document.getElementById('timerWarning');
const scratchLines = document.getElementById('scratchLines');
const victoryScreen = document.getElementById('victoryScreen');

// Game state
let currentCaptcha = '';
let successfulVerifications = 0;
let totalAttempts = 0;
let gameStartTime = null;
let level = 1;
let dissolveTimers = [];
let isDissolving = false;

// Character sets
const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const confusables = {
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'S': '5', '5': 'S',
    'B': '8', '8': 'B',
    'G': '6', '6': 'G',
    'Z': '2', '2': 'Z'
};

// Level configurations - dissolveDelay is time before first char starts fading
const levels = [
    { length: 5, dissolveDelay: 2500, dissolveSpeed: 1500, effects: ['fade'] },
    { length: 5, dissolveDelay: 2000, dissolveSpeed: 1200, effects: ['dissolve'] },
    { length: 6, dissolveDelay: 1800, dissolveSpeed: 1000, effects: ['dissolve', 'glitch'] },
    { length: 6, dissolveDelay: 1500, dissolveSpeed: 800, effects: ['dissolve', 'scramble'] },
    { length: 7, dissolveDelay: 1200, dissolveSpeed: 600, effects: ['dissolve', 'glitch', 'scramble'] }
];

// Error messages
const errorMessages = [
    "Incorrect. The characters shown were: [REDACTED]",
    "Verification failed. Are you sure you're human?",
    "Wrong. But at least you tried. Sort of.",
    "Nope. The correct answer has been destroyed for your protection.",
    "Error: Human verification inconclusive.",
    "Access denied. The text was right there. Mostly.",
    "Invalid input. The CAPTCHA has reported you to the authorities.",
    "Incorrect. Perhaps try glasses?",
    "Wrong answer. The CAPTCHA is judging you.",
    "Verification failed. The letters didn't like you either."
];

function init() {
    generateCaptcha();
    generateScratchLines();

    verifyBtn.addEventListener('click', verify);
    refreshBtn.addEventListener('click', refreshCaptcha);
    captchaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') verify();
    });

    // Auto-start dissolving after a brief moment
    setTimeout(startDissolving, 1000);
}

function generateCaptcha() {
    // Clear any existing timers
    dissolveTimers.forEach(t => clearTimeout(t));
    dissolveTimers = [];
    isDissolving = false;

    const config = levels[Math.min(level - 1, levels.length - 1)];
    currentCaptcha = '';

    for (let i = 0; i < config.length; i++) {
        currentCaptcha += chars[Math.floor(Math.random() * chars.length)];
    }

    renderCaptcha();
    captchaInput.value = '';
    feedback.className = 'feedback';
    timerWarning.classList.remove('visible');

    // Auto-start dissolving after a brief moment
    setTimeout(startDissolving, 500);
}

function renderCaptcha() {
    captchaText.innerHTML = '';

    for (let i = 0; i < currentCaptcha.length; i++) {
        const span = document.createElement('span');
        span.className = 'captcha-char';
        span.textContent = currentCaptcha[i];
        span.dataset.index = i;
        span.dataset.original = currentCaptcha[i];

        // Random rotation and position offset
        const rotation = (Math.random() - 0.5) * 20;
        const yOffset = (Math.random() - 0.5) * 10;
        span.style.transform = `rotate(${rotation}deg) translateY(${yOffset}px)`;

        captchaText.appendChild(span);
    }
}

function generateScratchLines() {
    scratchLines.innerHTML = '';

    for (let i = 0; i < 5; i++) {
        const line = document.createElement('div');
        line.className = 'scratch-line';
        line.style.top = (Math.random() * 100) + '%';
        line.style.left = '-10%';
        line.style.width = '120%';
        line.style.height = (1 + Math.random() * 2) + 'px';
        line.style.transform = `rotate(${(Math.random() - 0.5) * 10}deg)`;
        scratchLines.appendChild(line);
    }
}

function startDissolving() {
    if (isDissolving) return;
    if (!gameStartTime) gameStartTime = Date.now();

    isDissolving = true;
    const config = levels[Math.min(level - 1, levels.length - 1)];
    const chars = captchaText.querySelectorAll('.captcha-char');

    // Show warning shortly before dissolving starts
    const warningTime = Math.max(config.dissolveDelay - 1000, 500);
    setTimeout(() => {
        timerWarning.classList.add('visible');
    }, warningTime);

    // Schedule dissolving for each character
    chars.forEach((char, i) => {
        const delay = config.dissolveDelay + (i * 300);

        // Apply effects progressively
        const timer = setTimeout(() => {
            // Pick a random effect from available effects
            const effect = config.effects[Math.floor(Math.random() * config.effects.length)];

            if (effect === 'scramble') {
                // Rapidly change the character before fading
                scrambleCharacter(char);
            } else if (effect === 'glitch') {
                char.classList.add('glitching');
            }

            // Start the main dissolve/fade effect
            setTimeout(() => {
                char.classList.remove('glitching');
                if (config.effects.includes('dissolve')) {
                    char.classList.add('dissolving');
                } else {
                    char.classList.add('fading');
                }
            }, effect === 'glitch' ? 500 : 0);

        }, delay);

        dissolveTimers.push(timer);
    });
}

function scrambleCharacter(charElement) {
    let scrambleCount = 0;
    const maxScrambles = 10;
    const original = charElement.dataset.original;

    charElement.classList.add('scrambling');

    const scrambleInterval = setInterval(() => {
        // Show confusable or random character
        if (confusables[original] && Math.random() > 0.5) {
            charElement.textContent = confusables[original];
        } else {
            charElement.textContent = chars[Math.floor(Math.random() * chars.length)];
        }

        scrambleCount++;
        if (scrambleCount >= maxScrambles) {
            clearInterval(scrambleInterval);
            charElement.classList.remove('scrambling');
        }
    }, 100);
}

function verify() {
    totalAttempts++;
    const userInput = captchaInput.value.toUpperCase().trim();

    if (userInput === currentCaptcha) {
        successfulVerifications++;
        updateProgress();

        if (successfulVerifications >= 5) {
            showVictory();
        } else {
            feedback.textContent = "Verified! Loading next challenge...";
            feedback.className = 'feedback success';

            level++;
            difficultyBadge.textContent = `Level ${level}`;

            setTimeout(() => {
                generateCaptcha();
            }, 1500);
        }
    } else {
        feedback.textContent = errorMessages[Math.floor(Math.random() * errorMessages.length)];
        feedback.className = 'feedback error';

        // Generate new CAPTCHA after failure
        setTimeout(() => {
            generateCaptcha();
        }, 2000);
    }
}

function refreshCaptcha() {
    totalAttempts++;
    generateCaptcha();
    feedback.className = 'feedback';
    captchaInput.focus();
}

function updateProgress() {
    const progress = (successfulVerifications / 5) * 100;
    progressFill.style.width = progress + '%';
    attemptCounter.textContent = `${successfulVerifications} / 5 verified`;
}

function showVictory() {
    const endTime = Date.now();
    const totalTime = Math.round((endTime - gameStartTime) / 1000);
    const accuracy = Math.round((successfulVerifications / totalAttempts) * 100);

    victoryScreen.style.display = 'flex';
    victoryScreen.innerHTML = `
        <div class="victory-content">
            <h2>Human Verified!</h2>
            <p>Congratulations, you've proven you're not a robot. Probably.</p>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-value">${totalTime}s</div>
                    <div class="stat-label">Time</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${totalAttempts}</div>
                    <div class="stat-label">Attempts</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${accuracy}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
            <p style="font-size: 12px; color: #666; margin-top: 20px;">
                Your humanity score has been recorded and will be used against you in future verifications.
            </p>
            <div style="margin-top: 30px;">
                <button class="btn-restart" onclick="restartGame()">Try Again</button>
                <button class="btn-menu" onclick="window.location.href='../../index.html'">Back to Menu</button>
            </div>
        </div>
    `;
}

function restartGame() {
    victoryScreen.style.display = 'none';
    successfulVerifications = 0;
    totalAttempts = 0;
    gameStartTime = null;
    level = 1;

    difficultyBadge.textContent = 'Level 1';
    updateProgress();
    generateCaptcha();
}

window.restartGame = restartGame;

// Initialize
init();
