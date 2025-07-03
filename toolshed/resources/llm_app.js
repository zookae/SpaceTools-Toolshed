let sessionId = generateSessionId();
let uploadedImages = [];
let reasoningChain = [];
let currentProvider = 'openai';
let currentModel = '';
let currentTaskId = null;

document.getElementById('session-id').textContent = sessionId;

// Initialize
loadProviders();
loadTools();
loadVariables();
initializeSavedLinks();

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9);
}

function initializeSavedLinks() {
    // Set saved conversation links to use the current hostname instead of localhost
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;

    document.getElementById('logs-link').href = `${protocol}//${hostname}:9001`;
    document.getElementById('saved-link').href = `${protocol}//${hostname}:9002`;
}

function formatTextWithNewlines(text) {
    // Handle non-string inputs
    if (text === null || text === undefined) {
        return '';
    }

    // Convert objects to JSON strings
    if (typeof text === 'object') {
        text = JSON.stringify(text, null, 2);
    }

    // Convert to string if not already
    text = String(text);

    // Convert backslash-n to actual line breaks and escape HTML
    return text.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/\\n/g, '\n');
}

let originalToolStates = {};
let currentToolStates = {};

async function loadProviders() {
    try {
        const response = await fetch('/api/providers');
        const data = await response.json();

        const providerDropdown = document.getElementById('provider-dropdown');
        providerDropdown.innerHTML = data.providers.map(provider =>
            `<option value="${provider.id}" ${provider.id === data.current_provider ? 'selected' : ''}>
                ${provider.name} - ${provider.description}
            </option>`
        ).join('');

        currentProvider = data.current_provider;

        // Load models for current provider
        await loadModels();

    } catch (error) {
        console.error('Error loading providers:', error);
        showStatus('Error loading providers', 'error');
    }
}

async function changeProvider() {
    const providerDropdown = document.getElementById('provider-dropdown');
    const newProvider = providerDropdown.value;

    if (newProvider === currentProvider) {
        return;
    }

    // Show loading state
    showStatus('Changing provider...', 'info');
    disableControls(true);

    try {
        const formData = new FormData();
        formData.append('provider', newProvider);

        const response = await fetch('/api/provider/change', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            currentProvider = newProvider;

            // Reload models for new provider
            await loadModels();

            // Clear conversation since sessions are reset
            clearConversationUI();

            showStatus(`Switched to ${data.provider} provider`, 'success');
        } else {
            const error = await response.json();
            showStatus(`Failed to change provider: ${error.detail}`, 'error');
            // Revert selection
            providerDropdown.value = currentProvider;
        }
    } catch (error) {
        console.error('Error changing provider:', error);
        showStatus('Error changing provider', 'error');
        // Revert selection
        providerDropdown.value = currentProvider;
    } finally {
        disableControls(false);
    }
}

async function loadTools() {
    try {
        const response = await fetch('/api/tools');
        const data = await response.json();

        const toolsList = document.getElementById('tools-list');
        toolsList.innerHTML = data.tools.map(tool => {
            // Track original state
            originalToolStates[tool.name] = tool.enabled;
            currentToolStates[tool.name] = tool.enabled;

            return `
                <div class="tool-item ${!tool.enabled ? 'disabled' : ''}" id="tool-${tool.name}">
                    <div style="flex: 1;">
                        <div class="tool-header" onclick="toggleToolDescription('${tool.name}')">
                            <span class="tool-collapse-icon" id="collapse-icon-${tool.name}">▶</span>
                            <span class="tool-name">${tool.name}</span>
                        </div>
                        <div class="tool-description collapsed" id="description-${tool.name}">${tool.description}</div>
                    </div>
                    <div class="tool-toggle">
                        <input type="checkbox"
                               class="tool-checkbox"
                               id="toggle-${tool.name}"
                               ${tool.enabled ? 'checked' : ''}
                               onchange="toggleTool('${tool.name}')">
                        <label for="toggle-${tool.name}" style="cursor: pointer;">
                            ${tool.enabled ? 'Enabled' : 'Disabled'}
                        </label>
                    </div>
                </div>
            `;
        }).join('');

        // Show/hide variables tab based on configuration
        const variablesSection = document.getElementById('variables-section');
        if (variablesSection) {
            variablesSection.style.display = data.variables_enabled ? 'block' : 'none';
        }

    } catch (error) {
        console.error('Error loading tools:', error);
        document.getElementById('tools-list').innerHTML = '<p style="color: red;">Error loading tools</p>';
    }
}

function toggleToolDescription(toolName) {
    const description = document.getElementById(`description-${toolName}`);
    const icon = document.getElementById(`collapse-icon-${toolName}`);

    if (description.classList.contains('collapsed')) {
        description.classList.remove('collapsed');
        icon.textContent = '▼';
    } else {
        description.classList.add('collapsed');
        icon.textContent = '▶';
    }
}

async function toggleTool(toolName) {
    const checkbox = document.getElementById(`toggle-${toolName}`);
    const toolItem = document.getElementById(`tool-${toolName}`);
    const label = checkbox.nextElementSibling;

    currentToolStates[toolName] = checkbox.checked;

    if (checkbox.checked) {
        toolItem.classList.remove('disabled');
        label.textContent = 'Enabled';
    } else {
        toolItem.classList.add('disabled');
        label.textContent = 'Disabled';
    }

    // Apply changes immediately
    await applyToolConfiguration();
}

async function applyToolConfiguration() {
    try {
        showStatus('Applying tool configuration...', 'info');

        const formData = new FormData();
        formData.append('enabled_tools', JSON.stringify(currentToolStates));

        const response = await fetch('/api/tools/update', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            showStatus('Tool configuration updated successfully', 'success');

            // Update original states
            originalToolStates = {...currentToolStates};
        } else {
            showStatus('Failed to update tool configuration', 'error');
        }
    } catch (error) {
        console.error('Error updating tool configuration:', error);
        showStatus('Error updating tool configuration', 'error');
    }
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        const modelDropdown = document.getElementById('model-dropdown');
        modelDropdown.innerHTML = data.models.map(model =>
            `<option value="${model.id}" ${model.id === data.current_model ? 'selected' : ''}>
                ${model.name} - ${model.description}
            </option>`
        ).join('');

        currentModel = data.current_model;

        // Update current provider/model display
        document.getElementById('current-provider-model').textContent = `${data.current_provider} / ${data.current_model}`;

    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('model-dropdown').innerHTML = '<option>Error loading models</option>';
    }
}

async function changeModel() {
    const modelDropdown = document.getElementById('model-dropdown');
    const newModel = modelDropdown.value;

    if (newModel === currentModel || !newModel) {
        return;
    }

    showStatus('Changing model...', 'info');

    try {
        const formData = new FormData();
        formData.append('model', newModel);

        const response = await fetch('/api/model/change', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            currentModel = newModel;
            document.getElementById('current-provider-model').textContent = `${currentProvider} / ${currentModel}`;
            showStatus('Model changed successfully', 'success');
        } else {
            showStatus('Failed to change model', 'error');
            // Revert selection
            modelDropdown.value = currentModel;
        }
    } catch (error) {
        console.error('Error changing model:', error);
        showStatus('Error changing model', 'error');
        // Revert selection
        modelDropdown.value = currentModel;
    }
}

async function handleImageUpload() {
    const input = document.getElementById('image-upload');
    const files = input.files;

    if (files.length === 0) return;

    showStatus('Uploading images...', 'info');

    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        uploadedImages = data.images;

        // Display uploaded images
        const container = document.getElementById('uploaded-images');
        container.innerHTML = uploadedImages.map(img => {
            const imgEl = document.createElement('img');
            imgEl.src = 'data:image/jpeg;base64,' + img.data;
            imgEl.alt = img.filename;
            imgEl.style.maxWidth = '100px';
            imgEl.style.maxHeight = '100px';
            imgEl.style.margin = '5px';
            imgEl.style.cursor = 'pointer';
            imgEl.onclick = () => showImageModal('data:image/jpeg;base64,' + img.data);
            return imgEl.outerHTML;
        }).join('');

        showStatus(`Uploaded ${uploadedImages.length} image(s)`, 'success');

    } catch (error) {
        console.error('Error uploading images:', error);
        showStatus('Error uploading images', 'error');
    }
}

function showImageModal(imageSrc) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    modal.style.display = 'block';
    modalImg.src = imageSrc;
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function disableControls(disabled) {
    document.getElementById('message-input').disabled = disabled;
    document.getElementById('send-button').disabled = disabled;
    document.getElementById('image-upload').disabled = disabled;
}

function clearConversationUI() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = `
        <div class="message assistant-message">
            <strong>Assistant:</strong> Hello! I can help you analyze images using computer vision tools and execute Python code. Upload some images and ask me questions about them!
        </div>
    `;

    // Clear reasoning chain
    reasoningChain = [];
    updateReasoningDisplay();

    // Clear uploaded images
    uploadedImages = [];
    document.getElementById('uploaded-images').innerHTML = '';
    document.getElementById('image-upload').value = '';
}

async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    addMessage('user', message);
    input.value = '';

    // Show loading and create thinking steps container
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('send-button').disabled = true;

    // Update auto-save indicator to idle while processing
    updateAutoSaveIndicator('idle');

    // Create thinking steps container immediately
    const thinkingContainer = createThinkingStepsContainer();

    try {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('message', message);
        if (uploadedImages.length > 0) {
            formData.append('image_data', JSON.stringify(uploadedImages));
        }

        // Start the chat processing
        const response = await fetch('/api/chat/start', {
            method: 'POST',
            body: formData
        });

        const startData = await response.json();
        const taskId = startData.task_id;

        // Track current task for stop functionality
        currentTaskId = taskId;

        // Start polling for updates
        await pollForUpdates(taskId, thinkingContainer);

        // Refresh variables after processing
        await loadVariables();

        // Clear uploaded images after sending
        uploadedImages = [];
        document.getElementById('uploaded-images').innerHTML = '';
        document.getElementById('image-upload').value = '';

    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('assistant', 'Sorry, there was an error processing your request.');
        showStatus('Error sending message', 'error');
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('send-button').disabled = false;
        currentTaskId = null;
    }
}

async function stopProcessing() {
    if (!currentTaskId) {
        showStatus('No active processing to stop', 'info');
        return;
    }

    try {
        showStatus('Stopping AI processing...', 'info');

        const response = await fetch(`/api/chat/stop/${currentTaskId}`, {
            method: 'POST'
        });

        if (response.ok) {
            showStatus('Stop signal sent - AI will stop after current step', 'success');
        } else {
            const error = await response.json();
            showStatus(`Failed to stop: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error stopping processing:', error);
        showStatus('Error stopping processing', 'error');
    }
}

function createThinkingStepsContainer() {
    const messages = document.getElementById('messages');
    const stepsContainer = document.createElement('div');
    stepsContainer.className = 'thinking-steps';

    const headerDiv = document.createElement('div');
    headerDiv.style.padding = '10px 12px';
    headerDiv.style.fontWeight = 'bold';
    headerDiv.style.borderBottom = '2px solid #e2e8f0';
    headerDiv.style.background = '#f1f5f9';
    headerDiv.innerHTML = '🧠 AI Thinking Process';
    stepsContainer.appendChild(headerDiv);

    messages.appendChild(stepsContainer);
    messages.scrollTop = messages.scrollHeight;

    return stepsContainer;
}

async function pollForUpdates(taskId, thinkingContainer) {
    const maxPolls = 600; // 5 minutes max (increased for complex tasks)
    let pollCount = 0;
    let lastStepCount = 0;

    while (pollCount < maxPolls) {
        try {
            const response = await fetch(`/api/chat/status/${taskId}`);
            const data = await response.json();

            // Add new steps that we haven't seen yet
            if (data.steps && data.steps.length > lastStepCount) {
                const newSteps = data.steps.slice(lastStepCount);
                newSteps.forEach(step => addStepToContainer(thinkingContainer, step));
                lastStepCount = data.steps.length;
            }

            // Check if completed
            if (data.completed) {
                if (data.status === 'stopped') {
                    addMessage('assistant', data.response || 'Processing stopped by user.');
                    showStatus('Processing stopped', 'info');
                    updateAutoSaveIndicator('idle');
                } else if (data.error) {
                    addMessage('assistant', `Error: ${data.error}`);
                    showStatus('Error processing request', 'error');
                    updateAutoSaveIndicator('idle');
                } else {
                    addMessage('assistant', data.response);
                    if (data.tool_calls_made > 0) {
                        showStatus(`Used ${data.tool_calls_made} tool call(s) in ${data.iterations} iteration(s)`, 'info');
                    }

                    // Update auto-save indicator
                    if (data.auto_saved === true) {
                        updateAutoSaveIndicator('saved');
                    } else if (data.auto_saved === false) {
                        updateAutoSaveIndicator('failed');
                    }
                    // If undefined, leave it as idle (don't update)
                }
                break;
            }

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, 500));
            pollCount++;

        } catch (error) {
            console.error('Error polling for updates:', error);
            break;
        }
    }

    if (pollCount >= maxPolls) {
        addMessage('assistant', 'Request timed out after 5 minutes. The AI might be doing complex reasoning - please try again.');
        showStatus('Request timed out after 5 minutes', 'error');
    }
}

function addStepToContainer(container, step) {
    const stepDiv = document.createElement('div');
    stepDiv.className = `step ${step.type}`;
    const stepSpan = document.createElement('span');
    stepSpan.textContent = step.message;
    stepDiv.appendChild(stepSpan);

    // Add reasoning to chain of thought sidebar
    if (step.type === 'reasoning' && step.full_content) {
        addToReasoningChain(step);

        stepDiv.style.cursor = 'pointer';
        stepDiv.onclick = () => toggleStepDetails(stepDiv, step.full_content);

        // Add reasoning stats if available
        if (step.word_count || step.char_count) {
            const statsDiv = document.createElement('div');
            statsDiv.className = 'reasoning-stats';
            statsDiv.textContent = `${step.word_count || 0} words, ${step.char_count || 0} characters - Click to expand full reasoning`;
            stepDiv.appendChild(statsDiv);
        }

        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'step-details';
        const formattedContent = formatTextWithNewlines(step.full_content);
        detailsDiv.innerHTML = `<strong>Full AI Reasoning:</strong><br><br>${formattedContent}`;
        stepDiv.appendChild(detailsDiv);
    }

    if (step.type === 'tool_decision' && step.tool_calls) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'step-details expanded';

        step.tool_calls.forEach(tc => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'tool-call-details';

            // Format arguments and reasoning with proper newlines
            const formattedArgs = formatTextWithNewlines(tc.arguments);
            const formattedReasoning = formatTextWithNewlines(tc.reasoning);

            toolDiv.innerHTML = `
                <strong>${tc.name}</strong><br>
                Arguments: ${formattedArgs}<br>
                <em>${formattedReasoning}</em>
            `;
            detailsDiv.appendChild(toolDiv);
        });

        stepDiv.appendChild(detailsDiv);
    }

    if (step.type === 'tool_result' && step.full_result) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'step-details';
        const formattedResult = formatTextWithNewlines(step.full_result);
        let resultHtml = `<strong>Full Result:</strong><br>${formattedResult}`;
        detailsDiv.innerHTML = resultHtml;
        stepDiv.appendChild(detailsDiv);

        // If this step generated an image, display it inline
        if (step.has_image && step.image_data) {
            const imageContainer = document.createElement('div');
            imageContainer.className = 'tool-generated-image-container';

            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${step.image_data}`;
            img.className = 'tool-generated-image inline-image';
            img.onclick = () => showImageModal(img.src);

            const caption = document.createElement('div');
            caption.className = 'image-caption';
            caption.textContent = `Generated by ${step.tool_name} at ${new Date(step.timestamp).toLocaleTimeString()}`;

            imageContainer.appendChild(img);
            imageContainer.appendChild(caption);
            stepDiv.appendChild(imageContainer);
        }
    }

    container.appendChild(stepDiv);

    // Scroll to show new step
    const messages = document.getElementById('messages');
    messages.scrollTop = messages.scrollHeight;
}

function addToReasoningChain(step) {
    reasoningChain.push({
        iteration: step.iteration,
        content: step.full_content
    });
    updateReasoningDisplay();
}

function toggleStepDetails(stepDiv, content) {
    const detailsDiv = stepDiv.querySelector('.step-details');
    if (detailsDiv) {
        detailsDiv.classList.toggle('expanded');
    }
}

function openModal(imageSrc) {
    showImageModal(imageSrc);
}

function updateReasoningDisplay() {
    const reasoningDiv = document.getElementById('reasoning-chain');
    const countSpan = document.getElementById('reasoning-count');

    countSpan.textContent = `(${reasoningChain.length})`;

    if (reasoningChain.length === 0) {
        reasoningDiv.innerHTML = '<p style="font-size: 0.9em; color: #6b7280;">AI reasoning will appear here...</p>';
        return;
    }

    reasoningDiv.innerHTML = reasoningChain.map((reasoning, index) => `
        <div style="margin-bottom: 12px; padding: 8px; background: white; border-radius: 4px; border: 1px solid #e2e8f0;">
            <div style="font-size: 0.8em; color: #6b7280; margin-bottom: 4px;">
                Iteration ${reasoning.iteration}
            </div>
            <div style="font-size: 0.9em; white-space: pre-wrap;">
                ${formatTextWithNewlines(reasoning.content)}
            </div>
        </div>
    `).join('');

    // Scroll to bottom
    reasoningDiv.scrollTop = reasoningDiv.scrollHeight;
}

function addMessage(role, content) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const formattedContent = formatTextWithNewlines(content);
    messageDiv.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">${formattedContent}</pre>`;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';

    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
    }
}

function updateAutoSaveIndicator(status) {
    const indicator = document.getElementById('auto-save-indicator');
    if (!indicator) return;

    switch(status) {
        case 'saving':
            indicator.innerHTML = '🟡 Auto-saving...';
            indicator.style.color = '#f59e0b';
            break;
        case 'saved':
            indicator.innerHTML = '🟢 Auto-saved';
            indicator.style.color = '#10b981';
            break;
        case 'failed':
            indicator.innerHTML = '🔴 Auto-save failed';
            indicator.style.color = '#ef4444';
            break;
        case 'idle':
        default:
            indicator.innerHTML = '⚪ Auto-save idle';
            indicator.style.color = '#6b7280';
            break;
    }
}

async function saveConversation() {
    console.log('Save conversation button clicked, sessionId:', sessionId);

    const name = prompt('Enter a name for this conversation:');
    console.log('User entered name:', name);

    if (!name || name.trim() === '') {
        console.log('Name empty or cancelled, aborting save');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('name', name.trim());

        console.log(`Sending save request to /api/session/${sessionId}/save`);
        const response = await fetch(`/api/session/${sessionId}/save`, {
            method: 'POST',
            body: formData
        });

        console.log('Save response status:', response.status);

        if (response.ok) {
            const data = await response.json();
            console.log('Save successful:', data);
            showStatus(`✅ Conversation saved as "${name}"`, 'success');
        } else {
            const error = await response.json();
            console.error('Save failed:', error);
            showStatus(`Failed to save: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error saving conversation:', error);
        showStatus('Error saving conversation', 'error');
    }
}

async function clearSession() {
    if (!confirm('Are you sure you want to clear this conversation?')) {
        return;
    }

    try {
        const response = await fetch(`/api/session/${sessionId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Generate new session ID
            sessionId = generateSessionId();
            document.getElementById('session-id').textContent = sessionId;

            // Clear UI
            clearConversationUI();

            // Clear reasoning chain
            reasoningChain = [];
            updateReasoningDisplay();

            // Clear variables
            loadVariables();

            // Reset auto-save indicator
            updateAutoSaveIndicator('idle');

            showStatus('Conversation cleared', 'success');
        } else {
            showStatus('Failed to clear session', 'error');
        }
    } catch (error) {
        console.error('Error clearing session:', error);
        showStatus('Error clearing session', 'error');
    }
}

async function loadVariables() {
    try {
        const response = await fetch(`/api/session/${sessionId}/variables`);
        const data = await response.json();

        const variablesList = document.getElementById('variables-list');
        const variableCount = document.getElementById('variable-count');

        if (!data.variables || Object.keys(data.variables).length === 0) {
            variablesList.innerHTML = '<p style="font-size: 0.9em; color: #6b7280;">No variables stored yet</p>';
            variableCount.textContent = '(0)';
            return;
        }

        variableCount.textContent = `(${data.stats.total_variables})`;

        variablesList.innerHTML = Object.entries(data.variables).map(([name, value]) => {
            let displayValue = value;
            let variableType = typeof value;

            // Handle serialized complex objects
            if (typeof value === 'object' && value !== null && value._type) {
                // This is a serialized complex object
                variableType = value._type;
                displayValue = value._summary || JSON.stringify(value, null, 2);

                // Add additional info for specific types
                if (value._type === 'PIL_Image') {
                    displayValue = `${value._summary}\nDimensions: ${value.width}x${value.height}\nMode: ${value.mode}`;
                } else if (value._type === 'numpy_array') {
                    displayValue = `${value._summary}\nShape: [${value.shape.join(', ')}]\nSize: ${value.size} elements`;
                }
            } else {
                // Handle regular JSON-serializable values
                if (typeof value === 'object') {
                    displayValue = JSON.stringify(value, null, 2);
                } else if (typeof value === 'string' && value.length > 100) {
                    displayValue = value.substring(0, 100) + '...';
                }

                variableType = Array.isArray(value) ? 'array' : typeof value;
            }

            return `
                <div style="margin-bottom: 10px; padding: 8px; background: #f3f4f6; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <code style="font-weight: bold; color: #059669;">$${name}</code>
                        <span style="font-size: 0.8em; color: #6b7280;">${variableType}</span>
                    </div>
                    <pre style="margin-top: 4px; font-size: 0.8em; white-space: pre-wrap; overflow-wrap: break-word;">${displayValue}</pre>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error loading variables:', error);
    }
}

// Refresh variables every 5 seconds if there's an active session
setInterval(() => {
    if (sessionId) {
        loadVariables();
    }
}, 5000);

async function restartToolshed() {
    const enableVariables = document.getElementById('enableVariables').checked;
    const enableImages = document.getElementById('enableImages').checked;

    if (!confirm('⚠️ WARNING: This will restart toolshed for ALL users and clear all sessions. Continue?')) {
        return;
    }

    const button = document.getElementById('restartToolshed');
    const statusDiv = document.getElementById('configStatus');

    // Disable button and show loading
    button.disabled = true;
    button.innerHTML = '⏳ RESTARTING...';
    statusDiv.innerHTML = '<div class="loading-spinner">Restarting toolshed... This may take a moment.</div>';
    statusDiv.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('enable_variables', enableVariables);
        formData.append('enable_images', enableImages);

        const response = await fetch('/api/config/update', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();

            // Show success
            statusDiv.innerHTML = '<div class="success-message">✅ Toolshed restarted successfully!</div>';

            // Reload tools to reflect new configuration
            await loadTools();

            // Clear current conversation since sessions are cleared
            sessionId = generateSessionId();
            document.getElementById('session-id').textContent = sessionId;
            clearConversationUI();

            // Show final status
            showStatus('Toolshed restarted with new configuration', 'success');

            // Hide status after 3 seconds
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);

        } else {
            const error = await response.json();
            statusDiv.innerHTML = `<div class="error-message">❌ Failed to restart: ${error.detail}</div>`;
            showStatus('Failed to restart toolshed', 'error');
        }

    } catch (error) {
        console.error('Error restarting toolshed:', error);
        statusDiv.innerHTML = '<div class="error-message">❌ Error restarting toolshed</div>';
        showStatus('Error restarting toolshed', 'error');
    } finally {
        // Re-enable button
        button.disabled = false;
        button.innerHTML = '🔄 RESTART TOOLSHED';
    }
}

// Load current configuration on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/config/current');
        const config = await response.json();

        document.getElementById('enableVariables').checked = config.enable_variables;
        document.getElementById('enableImages').checked = config.enable_images;

    } catch (error) {
        console.error('Error loading configuration:', error);
    }
});

// System Prompt Editor Functions
async function toggleSystemPromptEditor() {
    const editor = document.getElementById('systemPromptEditor');
    const textarea = document.getElementById('systemPromptText');

    if (editor.style.display === 'none') {
        // Show editor and load current prompt
        editor.style.display = 'block';

        try {
            const response = await fetch('/api/system-prompt');
            const data = await response.json();
            textarea.value = data.system_prompt;
        } catch (error) {
            console.error('Error loading system prompt:', error);
            showStatus('Failed to load system prompt', 'error');
        }
    } else {
        // Hide editor
        editor.style.display = 'none';
    }
}

async function saveSystemPrompt() {
    const textarea = document.getElementById('systemPromptText');
    const systemPrompt = textarea.value;

    if (!confirm('Save this system prompt? It will apply to new sessions only.')) {
        return;
    }

    try {
        const formData = new FormData();
        formData.append('system_prompt', systemPrompt);

        const response = await fetch('/api/system-prompt/update', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            showStatus('System prompt saved successfully!', 'success');
        } else {
            const error = await response.json();
            showStatus(`Failed to save system prompt: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error saving system prompt:', error);
        showStatus('Error saving system prompt', 'error');
    }
}

async function resetSystemPrompt() {
    if (!confirm('Reset system prompt to default? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/api/system-prompt/reset', {
            method: 'POST'
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('systemPromptText').value = data.system_prompt;
            showStatus('System prompt reset to default', 'success');
        } else {
            const error = await response.json();
            showStatus(`Failed to reset system prompt: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error resetting system prompt:', error);
        showStatus('Error resetting system prompt', 'error');
    }
}
