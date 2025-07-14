document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const loadingSpinner = document.getElementById('loading-spinner');

    // This relative path is correct for Vercel's file-based routing.
    const BACKEND_URL = '/api/chat';

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        addMessage(userMessage, 'user');
        userInput.value = '';
        loadingSpinner.classList.remove('hidden');

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: userMessage })
            });

            if (!response.ok) {
                throw new Error(`Network response was not ok. Status: ${response.status}`);
            }

            const data = await response.json();
            addMessage(data.answer, 'ai');

        } catch (error) {
            console.error('Fetch Error:', error);
            addMessage('Sorry, something went wrong. Please try again.', 'ai');
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageElement.appendChild(paragraph);
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});