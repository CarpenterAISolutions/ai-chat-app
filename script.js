// script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const loadingSpinner = document.getElementById('loading-spinner');

    const BACKEND_URL = '/api/chat';

    let conversationHistory = [];

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        addMessage(userMessage, 'user');
        conversationHistory.push({ role: 'user', parts: [{ text: userMessage }] });

        userInput.value = '';
        loadingSpinner.classList.remove('hidden');

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history: conversationHistory })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An unknown error occurred.');
            }

            addMessage(data.answer, 'ai');
            conversationHistory.push({ role: 'model', parts: [{ text: data.answer }] });

        } catch (error) {
            console.error('Fetch Error:', error);
            addMessage(`Error: ${error.message}`, 'ai');
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