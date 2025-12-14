/**
 * Phone Support
 *
 * Verify your email address over the phone with a customer support agent
 * who reads confusing emails using an absurd phonetic alphabet.
 */

(function() {
    // DOM elements
    const startScreen = document.getElementById('startScreen');
    const gameArea = document.getElementById('gameArea');
    const gameOver = document.getElementById('gameOver');
    const victoryScreen = document.getElementById('victoryScreen');
    const gameOverReason = document.getElementById('gameOverReason');
    const holdIndicator = document.getElementById('holdIndicator');
    const conversationArea = document.getElementById('conversationArea');
    const agentText = document.getElementById('agentText');
    const agentName = document.getElementById('agentName');
    const speakingIndicator = document.getElementById('speakingIndicator');
    const emailInput = document.getElementById('emailInput');
    const replayBtn = document.getElementById('replayBtn');
    const submitBtn = document.getElementById('submitBtn');
    const callTimer = document.getElementById('callTimer');
    const verifiedCount = document.getElementById('verifiedCount');
    const attemptsCount = document.getElementById('attemptsCount');
    const replaysCount = document.getElementById('replaysCount');
    const patienceFill = document.getElementById('patienceFill');
    const attemptsDisplay = document.getElementById('attemptsDisplay');

    // Game state
    let gameRunning = false;
    let currentEmail = '';
    let verified = 0;
    let attempts = 0;
    let replays = 0;
    let patience = 100;
    let callSeconds = 0;
    let callInterval = null;
    let currentEmailAttempts = 0;
    let speechSynth = window.speechSynthesis;
    let currentUtterance = null;
    let gameStartTime = 0;

    // Agent names
    const agentNames = ['Karen', 'Brenda', 'Linda', 'Deborah', 'Patricia', 'Cheryl'];

    // Confusing email addresses - designed to be hard to hear
    const emails = [
        'lI1lI1l@gmail.com',
        'O0O0O0o@yahoo.com',
        'underscore_not-hyphen@mail.com',
        'brian_not_ryan@aol.com',
        'mn.nm.mn@outlook.com',
        'dbd.bdb@icloud.com',
        'fifteen50fifty15@gmail.com',
        's5z2.2z5s@yahoo.com',
        'vv_w_vv@hotmail.com',
        'q9g.g9q@mail.com',
        'zero0O.letterO0@gmail.com',
        'one1l.lowercase_L@yahoo.com',
        'bee_b.sea_c@aol.com',
        'ess_s.eff_f@outlook.com',
        'CaseSensitive@Gmail.com',
        'dotdot..double@mail.com',
        'plus+sign@gmail.com',
        'the_real_john.smith.jr@yahoo.com',
        'xXx_epic_xXx@hotmail.com',
        'n0t.a.r0b0t@gmail.com',
        // Inspired by McSweeney's
        'MikeUnderscore2004@yahoo.com',
        'MikeAtYahooDotCom@hotmail.com',
        'Mike_WardAllOneWord@yahoo.com',
        'HyphenNotUnderscore-Jones@gmail.com',
        'NoSpacesExceptThisOne @mail.com',
        'Period.Period.Period@outlook.com',
        'IsItAnIOrA1@yahoo.com',
        'TwoTs_TwoRs_Garrett@aol.com',
        'SmithWithAnI_NotAY@gmail.com',
        'ThatsCapitalBNotD@hotmail.com',
        'AAAAAThatsSixAs@yahoo.com',
        'Lllll_FiveLowerCaseLs@gmail.com',
        'Eight8s88888888@hotmail.com',
        'MyLastNameIsNull@outlook.com',
        'Robert_DropTable@students.edu',
        'DotComNotDotNet.net@mail.com'
    ];

    // Absurd phonetic alphabet
    const phoneticAlphabet = {
        'a': ['A as in Aisle', 'A as in Aesthetic', 'A as in... Apple, I think'],
        'b': ['B as in Bdellium', 'B as in... uh... Boy', 'B, the letter after A'],
        'c': ['C as in Czar', 'C as in Cue... wait, that\'s Q', 'C as in Sea... no wait'],
        'd': ['D as in Django', 'D as in the letter D', 'D as in... Dog, sure'],
        'e': ['E as in Eye... no wait', 'E as in Euphemism', 'E, the vowel'],
        'f': ['F as in... hold on... Phantom', 'F as in the F word... I mean Fox', 'F as in Phf... no, F'],
        'g': ['G as in Gnat', 'G as in Gnome', 'G as in... Gee whiz'],
        'h': ['H as in Heir', 'H as in Hour', 'H as in... the silent one'],
        'i': ['I as in... wait, uppercase or lowercase?', 'I as in Eye', 'I, the letter, not the word'],
        'j': ['J as in Jalapeño', 'J as in... José', 'J, which sounds like G'],
        'k': ['K as in Knight', 'K as in Knife', 'K as in Know'],
        'l': ['L as in Fifty... wait no', 'L as in... Elle?', 'L, lowercase, not capital I'],
        'm': ['M as in Mnemonic', 'M as in... hmm', 'M or N? M.'],
        'n': ['N as in... November? Sure', 'N, not M', 'N as in Gnocchi... wait'],
        'o': ['O as in Opossum', 'O as in... the letter, not zero', 'O, the round one'],
        'p': ['P as in Pneumonia', 'P as in Pterodactyl', 'P as in Psychology'],
        'q': ['Q as in Queue', 'Q as in Quiche', 'Q, with the little tail'],
        'r': ['R as in... are you there?', 'R as in Rhetoric', 'R, the pirate letter'],
        's': ['S as in Sea... C? No, S', 'S as in... Snake', 'S, the curvy one'],
        't': ['T as in Tsunami', 'T as in Tsar', 'T as in... Tea? T.'],
        'u': ['U as in... you? No, U', 'U as in Ugh', 'U, not W'],
        'v': ['V as in... five in Roman', 'V as in Vee', 'V, the pointy one'],
        'w': ['W as in Write', 'W as in Whole', 'W as in Why'],
        'x': ['X as in Xylophone, obviously', 'X as in... the unknown', 'X marks the spot'],
        'y': ['Y as in... why? Because.', 'Y as in Yves', 'Y, sometimes a vowel'],
        'z': ['Z as in... Zed or Zee?', 'Z as in Zebra', 'Z, the last one'],
        '0': ['Zero, the number, not the letter O', 'Zero, like nothing', 'Zero, round but a number'],
        '1': ['One, the number, not lowercase L', 'One, a single digit', 'One, not the letter I'],
        '2': ['Two, the number', 'Two, like a pair', 'Two, T-W-O... wait you don\'t need to spell it'],
        '3': ['Three', 'Three, like a crowd', 'Three, the one after two'],
        '4': ['Four, F-O-U-R... I mean, just four', 'Four', 'Four, not for'],
        '5': ['Five, the number, not S', 'Five, like a hand', 'Five, halfway to ten'],
        '6': ['Six, the number', 'Six, S-I-X', 'Six, not nine upside down... well'],
        '7': ['Seven, lucky number', 'Seven', 'Seven, the one with the line'],
        '8': ['Eight, like the shape', 'Eight, E-I-G-H-T', 'Eight, infinity sideways... no wait'],
        '9': ['Nine, the number', 'Nine, almost ten', 'Nine, not six upside down'],
        '.': ['Dot', 'Period', 'Full stop... actually just dot'],
        '@': ['At sign', 'At symbol', 'The swirly A thing... at'],
        '_': ['Underscore, the low line', 'Underscore, not hyphen', 'The line at the bottom, underscore'],
        '-': ['Hyphen, not underscore', 'Dash, the middle one', 'Minus sign... hyphen'],
        '+': ['Plus sign', 'Plus, the cross', 'Plus symbol, like addition']
    };

    // Agent responses
    const agentResponses = {
        greeting: [
            "Thank you for calling GlobalTech support. My name is {agent}. I'll need to verify your email address.",
            "GlobalTech support, this is {agent}. For security purposes, I need to confirm your email.",
            "Hi, you've reached {agent} at GlobalTech. Let me verify your email on file."
        ],
        readingEmail: [
            "Okay, so I have your email as... let me see here...",
            "According to our system, your email is...",
            "I'm showing your email address as..."
        ],
        askConfirm: [
            "Can you confirm that's correct?",
            "Is that right?",
            "Does that match what you have?"
        ],
        wrongAnswer: [
            "Hmm, that doesn't match what I have. Let me read it again.",
            "No, that's not what I said. Listen carefully this time.",
            "That's incorrect. I'll repeat it, please pay attention.",
            "Are you even listening? Let me say it again.",
            "Wrong. Here it is again. Slower this time."
        ],
        impatient: [
            "Sir... Ma'am... we've been over this.",
            "I don't have all day here.",
            "There are other customers waiting, you know.",
            "This is a simple email verification.",
            "*sigh* One more time."
        ],
        correct: [
            "Yes! That's correct. Let me verify the next one.",
            "Finally. Okay, moving on.",
            "Correct. Next email.",
            "That's right. Verifying..."
        ],
        hangingUp: [
            "I'm sorry, I can't spend any more time on this. *click*",
            "Sir, I'm going to have to end this call. *click*",
            "I've repeated this too many times. Goodbye. *click*",
            "Call back when you're ready to listen. *click*"
        ],
        replay: [
            "Fine. One more time...",
            "*sigh* Again?",
            "I already said this but okay...",
            "Pay attention this time..."
        ]
    };

    // Get random item from array
    function randomFrom(arr) {
        return arr[Math.floor(Math.random() * arr.length)];
    }

    // Get phonetic for a character
    function getPhonetic(char) {
        const lower = char.toLowerCase();
        if (phoneticAlphabet[lower]) {
            return randomFrom(phoneticAlphabet[lower]);
        }
        return char;
    }

    // Build speech text for an email
    function buildEmailSpeech(email, style = 'normal') {
        let parts = [];

        for (let i = 0; i < email.length; i++) {
            const char = email[i];

            if (style === 'fast') {
                // Just say the character
                if (char === '@') parts.push('at');
                else if (char === '.') parts.push('dot');
                else if (char === '_') parts.push('underscore');
                else if (char === '-') parts.push('hyphen');
                else if (char === '+') parts.push('plus');
                else parts.push(char);
            } else if (style === 'phonetic') {
                // Use absurd phonetic alphabet
                parts.push(getPhonetic(char));
            } else {
                // Mix it up
                if (Math.random() > 0.6) {
                    parts.push(getPhonetic(char));
                } else {
                    if (char === '@') parts.push('at');
                    else if (char === '.') parts.push('dot');
                    else if (char === '_') parts.push('underscore');
                    else if (char === '-') parts.push('hyphen');
                    else if (char === '+') parts.push('plus');
                    else parts.push(char);
                }
            }
        }

        return parts.join(', ');
    }

    // Voice cache
    let selectedVoice = null;

    // Find the best available voice
    function getBestVoice() {
        if (selectedVoice) return selectedVoice;

        const voices = speechSynth.getVoices();
        if (voices.length === 0) return null;

        // Priority list - prefer natural/enhanced voices (macOS has good ones)
        const priorities = [
            // macOS premium voices
            v => v.name.includes('Samantha') && v.name.includes('Enhanced'),
            v => v.name.includes('Karen') && v.name.includes('Enhanced'),
            v => v.name.includes('Allison') && v.name.includes('Enhanced'),
            v => v.name.includes('Ava') && v.name.includes('Enhanced'),
            v => v.name.includes('Susan') && v.name.includes('Enhanced'),
            v => v.name.includes('Zoe') && v.name.includes('Enhanced'),
            // Any enhanced English voice
            v => v.name.includes('Enhanced') && v.lang.startsWith('en'),
            // Premium tier
            v => v.name.includes('Premium') && v.lang.startsWith('en'),
            // Specific good voices
            v => v.name === 'Samantha',
            v => v.name === 'Karen',
            v => v.name === 'Allison',
            // Google voices (Chrome)
            v => v.name.includes('Google') && v.lang.startsWith('en-US'),
            // Any US English
            v => v.lang === 'en-US',
            // Any English
            v => v.lang.startsWith('en')
        ];

        for (const test of priorities) {
            const voice = voices.find(test);
            if (voice) {
                selectedVoice = voice;
                console.log('Selected voice:', voice.name);
                return voice;
            }
        }

        selectedVoice = voices[0];
        return voices[0];
    }

    // Speak text
    function speak(text, options = {}) {
        return new Promise((resolve) => {
            if (currentUtterance) {
                speechSynth.cancel();
            }

            const utterance = new SpeechSynthesisUtterance(text);
            currentUtterance = utterance;

            // Get best voice
            const voice = getBestVoice();
            if (voice) {
                utterance.voice = voice;
            }

            // Natural variation in speech
            const baseRate = options.rate || 1.0;
            const basePitch = options.pitch || 1.0;

            // Add slight randomness for more natural feel
            utterance.rate = baseRate + (Math.random() * 0.1 - 0.05);
            utterance.pitch = basePitch + (Math.random() * 0.1 - 0.05);
            utterance.volume = options.volume || 1.0;

            speakingIndicator.classList.remove('silent');

            utterance.onend = () => {
                speakingIndicator.classList.add('silent');
                currentUtterance = null;
                resolve();
            };

            utterance.onerror = () => {
                speakingIndicator.classList.add('silent');
                currentUtterance = null;
                resolve();
            };

            speechSynth.speak(utterance);
        });
    }

    // Stop speaking
    function stopSpeaking() {
        if (speechSynth) {
            speechSynth.cancel();
        }
        speakingIndicator.classList.add('silent');
    }

    // Update call timer
    function updateCallTimer() {
        callSeconds++;
        const mins = Math.floor(callSeconds / 60);
        const secs = callSeconds % 60;
        callTimer.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    // Update stats display
    function updateStats() {
        verifiedCount.textContent = verified;
        attemptsCount.textContent = attempts;
        replaysCount.textContent = replays;
        patienceFill.style.width = patience + '%';

        if (currentEmailAttempts > 0) {
            attemptsDisplay.textContent = `Attempts for this email: ${currentEmailAttempts}`;
        } else {
            attemptsDisplay.textContent = '';
        }
    }

    // Reduce patience
    function reducePatience(amount) {
        patience = Math.max(0, patience - amount);
        updateStats();

        if (patience <= 0) {
            endGame(randomFrom(agentResponses.hangingUp));
        }
    }

    // Select a new email
    function selectNewEmail() {
        currentEmail = randomFrom(emails);
        currentEmailAttempts = 0;
        updateStats();
    }

    // Read the current email
    async function readEmail(isReplay = false) {
        replayBtn.disabled = true;
        submitBtn.disabled = true;

        if (isReplay) {
            const replayText = randomFrom(agentResponses.replay);
            agentText.textContent = replayText;
            await speak(replayText, { rate: 1.1 });
            await new Promise(r => setTimeout(r, 500));
        }

        // Determine style based on patience and attempts
        let style = 'normal';
        let rate = 1.0;

        if (patience < 50 || currentEmailAttempts > 2) {
            style = 'fast';
            rate = 1.3;
        } else if (currentEmailAttempts === 0) {
            style = 'phonetic';
            rate = 0.9;
        }

        const intro = randomFrom(agentResponses.readingEmail);
        agentText.textContent = intro;
        await speak(intro, { rate: 1.0 });

        await new Promise(r => setTimeout(r, 300));

        const emailSpeech = buildEmailSpeech(currentEmail, style);
        agentText.innerHTML = `"<span class="highlight">${currentEmail}</span>"`;
        await speak(emailSpeech, { rate: rate });

        await new Promise(r => setTimeout(r, 500));

        const confirm = randomFrom(agentResponses.askConfirm);
        agentText.innerHTML += `<br><br>${confirm}`;
        await speak(confirm, { rate: 1.0 });

        replayBtn.disabled = false;
        submitBtn.disabled = false;
        emailInput.focus();
    }

    // Replay email
    async function replayEmail() {
        if (!gameRunning) return;

        replays++;
        reducePatience(10);
        updateStats();

        if (patience > 0) {
            await readEmail(true);
        }
    }

    // Submit email
    async function submitEmail() {
        if (!gameRunning) return;

        const guess = emailInput.value.trim();
        attempts++;
        currentEmailAttempts++;
        updateStats();

        if (guess.toLowerCase() === currentEmail.toLowerCase()) {
            // Correct!
            emailInput.classList.add('success');
            verified++;
            updateStats();

            const correctText = randomFrom(agentResponses.correct);
            agentText.textContent = correctText;
            await speak(correctText, { rate: 1.0 });

            emailInput.value = '';
            emailInput.classList.remove('success');

            if (verified >= 3) {
                showVictory();
            } else {
                await new Promise(r => setTimeout(r, 1000));
                selectNewEmail();
                await readEmail();
            }
        } else {
            // Wrong
            emailInput.classList.add('error');
            setTimeout(() => emailInput.classList.remove('error'), 300);

            reducePatience(15);

            if (patience > 0) {
                let response;
                if (patience < 30) {
                    response = randomFrom(agentResponses.impatient);
                } else {
                    response = randomFrom(agentResponses.wrongAnswer);
                }

                agentText.textContent = response;
                await speak(response, { rate: 1.1 });

                await new Promise(r => setTimeout(r, 500));
                await readEmail();
            }
        }
    }

    // Start game
    window.startGame = async function() {
        startScreen.style.display = 'none';
        gameOver.style.display = 'none';
        victoryScreen.style.display = 'none';
        gameArea.style.display = 'block';
        conversationArea.style.display = 'block';
        holdIndicator.classList.remove('active');

        // Reset state
        gameRunning = true;
        verified = 0;
        attempts = 0;
        replays = 0;
        patience = 100;
        callSeconds = 0;
        currentEmailAttempts = 0;
        gameStartTime = Date.now();
        emailInput.value = '';

        // Set random agent name
        agentName.textContent = randomFrom(agentNames);

        updateStats();

        // Start call timer
        callInterval = setInterval(updateCallTimer, 1000);

        // Initial greeting
        const greeting = randomFrom(agentResponses.greeting).replace('{agent}', agentName.textContent);
        agentText.textContent = greeting;
        await speak(greeting, { rate: 0.95 });

        // Select first email and read it
        await new Promise(r => setTimeout(r, 500));
        selectNewEmail();
        await readEmail();
    };

    // End game
    function endGame(reason) {
        gameRunning = false;
        stopSpeaking();

        if (callInterval) {
            clearInterval(callInterval);
            callInterval = null;
        }

        gameArea.style.display = 'none';
        gameOver.style.display = 'block';
        gameOverReason.textContent = reason;
    }

    // Show victory
    function showVictory() {
        gameRunning = false;
        stopSpeaking();

        if (callInterval) {
            clearInterval(callInterval);
            callInterval = null;
        }

        const timeSpent = Math.floor((Date.now() - gameStartTime) / 1000);
        const mins = Math.floor(timeSpent / 60);
        const secs = timeSpent % 60;

        gameArea.style.display = 'none';
        victoryScreen.style.display = 'block';
        victoryScreen.innerHTML = `
            <h2>VERIFIED!</h2>
            <p>Your email addresses have been confirmed.<br>Thank you for your patience.</p>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div>
                    <div style="font-size: 24px; color: #0f0;">${mins}:${secs.toString().padStart(2, '0')}</div>
                    <div style="font-size: 11px; color: #666;">CALL TIME</div>
                </div>
                <div>
                    <div style="font-size: 24px; color: #0f0;">${attempts}</div>
                    <div style="font-size: 11px; color: #666;">ATTEMPTS</div>
                </div>
                <div>
                    <div style="font-size: 24px; color: #0f0;">${replays}</div>
                    <div style="font-size: 11px; color: #666;">REPLAYS</div>
                </div>
            </div>

            <div style="background: #1a1a2a; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: left;">
                <h3 style="color: #fff; font-size: 14px; margin-bottom: 10px;">The Hostile Mechanics</h3>
                <ul style="color: #999; font-size: 12px; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Confusing Characters:</strong> l/1/I, O/0, m/n, b/d</li>
                    <li><strong>Absurd Phonetic Alphabet:</strong> "P as in Pterodactyl"</li>
                    <li><strong>Speed Changes:</strong> Agent talks faster when impatient</li>
                    <li><strong>Patience Drain:</strong> Each mistake or replay costs patience</li>
                    <li><strong>Case Sensitivity:</strong> Sometimes it matters</li>
                </ul>
            </div>

            <button class="btn-start" onclick="startGame()">CALL AGAIN</button>
            <button class="btn-start" style="background: #333; color: #fff; margin-top: 10px;" onclick="location.href='../../index.html'">BACK TO MENU</button>
        `;
    }

    // Handle Enter key
    emailInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !submitBtn.disabled) {
            submitEmail();
        }
    });

    // Button click handlers
    submitBtn.addEventListener('click', submitEmail);
    replayBtn.addEventListener('click', replayEmail);

    // Load voices (needed for some browsers)
    if (speechSynth) {
        speechSynth.getVoices();
        speechSynth.onvoiceschanged = () => speechSynth.getVoices();
    }

    // Initialize
    speakingIndicator.classList.add('silent');
})();
