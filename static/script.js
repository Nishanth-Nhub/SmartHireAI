document.addEventListener('DOMContentLoaded', () => {
    // Navbar Auth Buttons
    const authButtons = document.getElementById('auth-buttons');
    if (authButtons) {
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        const isUploadPage = window.location.pathname === '/upload';
        if (currentUser) {
            // Show Logout button
            authButtons.innerHTML = `
                <button class="logout" id="logout-button">
                    <i class="fas fa-sign-out-alt"></i> Logout
                </button>
            `;
            const logoutButton = document.getElementById('logout-button');
            if (logoutButton) {
                logoutButton.addEventListener('click', () => {
                    localStorage.removeItem('currentUser');
                    alert('Logged out successfully!');
                    window.location.href = '/';
                });
            }
        } else if (!isUploadPage) {
            // Show Sign In and Register buttons, except on upload page
            authButtons.innerHTML = `
                <button class="cta-button sign-in" onclick="window.location.href='/login'">
                    <i class="fas fa-sign-in-alt"></i> Sign In
                </button>
                <button class="cta-button register" onclick="window.location.href='/register'">
                    <i class="fas fa-user-plus"></i> Register
                </button>
            `;
        }
    }

    // FAQ Accordion
    const faqQuestions = document.querySelectorAll('.faq-question');
    faqQuestions.forEach(question => {
        question.addEventListener('click', () => {
            const answer = question.nextElementSibling;
            const isActive = question.classList.contains('active');

            // Close all other FAQs
            document.querySelectorAll('.faq-question').forEach(q => {
                q.classList.remove('active');
                q.nextElementSibling.classList.remove('active');
            });

            // Toggle current FAQ
            if (!isActive) {
                question.classList.add('active');
                answer.classList.add('active');
            }
        });
    });

    const backToTopButton = document.getElementById('back-to-top');

    if (backToTopButton) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        });

        backToTopButton.addEventListener('click', (e) => {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    } else {
        console.warn("Back to Top button not found in the DOM.");
    }

    // Login Form
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const emailInput = document.getElementById('email');
            const passwordInput = document.getElementById('password');
            const email = emailInput.value.trim();
            const password = passwordInput.value.trim();

            // Reset invalid styles
            emailInput.classList.remove('invalid');
            passwordInput.classList.remove('invalid');

            // Basic validation
            if (!email) {
                emailInput.classList.add('invalid');
                alert('Please enter your email.');
                return;
            }
            if (!password) {
                passwordInput.classList.add('invalid');
                alert('Please enter your password.');
                return;
            }

            const users = JSON.parse(localStorage.getItem('users') || '[]');
            const user = users.find(u => u.email === email && u.password === password);
            if (user) {
                localStorage.setItem('currentUser', JSON.stringify(user));
                alert('Login successful!');
                window.location.href = '/';
            } else {
                emailInput.classList.add('invalid');
                passwordInput.classList.add('invalid');
                alert('Invalid email or password.');
            }
        });
    }

    // Forgot Password
    const forgotPasswordLink = document.querySelector('.forgot-password a');
    if (forgotPasswordLink) {
        forgottenPasswordLink.addEventListener('click', (e) => {
            e.preventDefault();
            alert('Forgot Password feature is under development. Please contact support@smarthireai.com.');
        });
    }

    // Register Form
    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const emailInput = document.getElementById('email');
            const passwordInput = document.getElementById('password');
            const confirmPasswordInput = document.getElementById('confirm-password');
            const email = emailInput.value.trim();
            const password = passwordInput.value.trim();
            const confirmPassword = confirmPasswordInput.value.trim();

            // Reset invalid styles
            emailInput.classList.remove('invalid');
            passwordInput.classList.remove('invalid');
            confirmPasswordInput.classList.remove('invalid');

            // Validation
            if (!email) {
                emailInput.classList.add('invalid');
                alert('Please enter your email.');
                return;
            }
            if (!password) {
                passwordInput.classList.add('invalid');
                alert('Please enter your password.');
                return;
            }
            if (password !== confirmPassword) {
                passwordInput.classList.add('invalid');
                confirmPasswordInput.classList.add('invalid');
                alert('Passwords do not match.');
                return;
            }

            const users = JSON.parse(localStorage.getItem('users') || '[]');
            if (users.some(u => u.email === email)) {
                emailInput.classList.add('invalid');
                alert('Email already registered.');
                return;
            }
            users.push({ email, password });
            localStorage.setItem('users', JSON.stringify(users));
            alert('Registration successful! Please sign in.');
            window.location.href = '/login';
        });
    }

    // Upload Form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('resume');
            const loading = document.getElementById('loading');
            if (!fileInput.files.length) {
                alert('Please select a PDF file.');
                return;
            }
            const file = fileInput.files[0];
            if (file.type !== 'application/pdf') {
                alert('Only PDF files are supported.');
                return;
            }

            loading.style.display = 'block';
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload_resume', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }
                const data = await response.json();
                if (data.redirect) {
                    window.location.href = data.redirect;
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert(`Upload failed: ${error.message}`);
                loading.style.display = 'none';
            }
        });
    }

    // Chatbot Form
    const chatForm = document.getElementById('chat-form');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const newChatButton = document.getElementById('new-chat');
    if (chatForm && chatMessages && chatInput) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;

            // Add user message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.textContent = message;
            chatMessages.appendChild(userDiv);
            chatInput.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Send to server
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Chat error: ${response.status} - ${errorText}`);
                }
                const data = await response.json();
                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-message';
                botDiv.innerHTML = marked.parse(data.message);
                chatMessages.appendChild(botDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } catch (error) {
                console.error('Chat error:', error);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message bot-message';
                errorDiv.textContent = `Sorry, something went wrong: ${error.message}`;
                chatMessages.appendChild(errorDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        });

        if (newChatButton) {
            newChatButton.addEventListener('click', () => {
                chatMessages.innerHTML = '';
            });
        }
    }
});