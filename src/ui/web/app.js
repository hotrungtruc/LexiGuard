/**
 * LexiGuard AI — Web Application Controller (Gemini & ChatGPT Inspired)
 * Multi-Agent Legal Assistant & Contract Vulnerability Auditor
 */

(() => {
  'use strict';

  // =========================================================================
  // CONSTANTS & STORAGE KEYS
  // =========================================================================
  const STORAGE_KEYS = {
    SESSIONS: 'lexiguard_sessions_v2',
    CURRENT_SESSION_ID: 'lexiguard_current_session_id',
    THEME: 'lexiguard_theme',
    OCR_VAULT: 'lexiguard_ocr_vault',
  };

  const STEP_METADATA = {
    init: { label: 'Khởi tạo quy trình thẩm định...', icon: '⚙️' },
    extract_clauses: { label: 'Trích xuất & chuẩn hoá cấu trúc điều khoản', icon: '📑' },
    retrieve_statutes: { label: 'Truy vấn đồ thị tri thức & kiểm toán bảng tính (GraphRAG)', icon: '⚖️' },
    audit_vulnerabilities: { label: 'Phân tích lỗ hổng pháp lý & đề xuất Redline (LoopholeAuditor)', icon: '🔍' },
    grade_hallucination: { label: 'Chấm điểm đối soát luật định & kiểm tra ảo giác', icon: '🛡️' },
    search: { label: 'Truy vấn kho văn bản luật định', icon: '🔎' },
    retrieved: { label: 'Kết nối văn bản & án lệ liên quan', icon: '📚' },
    synthesizing: { label: 'Tổng hợp luận cứ & soạn thảo câu trả lời', icon: '✍️' },
  };

  // =========================================================================
  // APPLICATION STATE
  // =========================================================================
  let state = {
    sessions: [],
    currentSessionId: null,
    attachedFile: null,
    isGenerating: false,
    activeController: null,
    ocrVault: [],
    theme: 'dark-theme',
  };

  // =========================================================================
  // DOM ELEMENTS
  // =========================================================================
  const dom = {
    // Sidebar
    sidebar: document.getElementById('sidebar'),
    sidebarOpenBtn: document.getElementById('sidebar-open-btn'),
    sidebarCloseBtn: document.getElementById('sidebar-close-btn'),
    newChatBtn: document.getElementById('new-chat-btn'),
    sessionSearchInput: document.getElementById('session-search-input'),
    clearSearchBtn: document.getElementById('clear-search-btn'),
    sessionsToday: document.getElementById('sessions-today'),
    sessionsRecent: document.getElementById('sessions-recent'),
    documentVaultList: document.getElementById('document-vault-list'),
    themeToggleBtn: document.getElementById('theme-toggle-btn'),
    clearAllSessionsBtn: document.getElementById('clear-all-sessions-btn'),
    systemStatus: document.getElementById('system-status'),

    // Chat Header & Viewport
    currentChatTitle: document.getElementById('current-chat-title'),
    openDrawerBtn: document.getElementById('open-drawer-btn'),
    exportChatBtn: document.getElementById('export-chat-btn'),
    chatViewport: document.getElementById('chat-viewport'),
    heroContainer: document.getElementById('hero-container'),
    messagesContainer: document.getElementById('messages-container'),
    promptCards: document.querySelectorAll('.prompt-card'),
    triggerOcrCard: document.getElementById('trigger-ocr-card'),

    // Prompt Bar
    attachmentsPreviewBar: document.getElementById('attachments-preview-bar'),
    fileUploadInput: document.getElementById('file-upload-input'),
    ocrUploadInput: document.getElementById('ocr-upload-input'),
    attachFileBtn: document.getElementById('attach-file-btn'),
    ocrBtn: document.getElementById('ocr-btn'),
    promptInput: document.getElementById('prompt-input'),
    sendBtn: document.getElementById('send-btn'),
    sendIcon: document.getElementById('send-icon'),
    stopIcon: document.getElementById('stop-icon'),

    // OCR Drawer
    drawerOverlay: document.getElementById('drawer-overlay'),
    documentDrawer: document.getElementById('document-drawer'),
    drawerCloseBtn: document.getElementById('drawer-close-btn'),
    drawerDocName: document.getElementById('drawer-doc-name'),
    drawerPageCount: document.getElementById('drawer-page-count'),
    drawerTableCount: document.getElementById('drawer-table-count'),
    drawerCharCount: document.getElementById('drawer-char-count'),
    drawerMarkdownPreview: document.getElementById('drawer-markdown-preview'),
    copyOcrTextBtn: document.getElementById('copy-ocr-text-btn'),
    sendOcrToChatBtn: document.getElementById('send-ocr-to-chat-btn'),
  };

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  function init() {
    loadTheme();
    loadStorage();
    bindEvents();
    renderSidebarSessions();
    renderDocumentVault();

    if (!state.currentSessionId || !getSessionById(state.currentSessionId)) {
      createNewSession(false);
    } else {
      loadSession(state.currentSessionId);
    }

    checkBackendHealth();
  }

  // =========================================================================
  // STORAGE & STATE MANAGEMENT
  // =========================================================================
  function loadStorage() {
    try {
      const storedSessions = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      state.sessions = storedSessions ? JSON.parse(storedSessions) : [];

      const storedVault = localStorage.getItem(STORAGE_KEYS.OCR_VAULT);
      state.ocrVault = storedVault ? JSON.parse(storedVault) : [];

      const storedSessionId = localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION_ID);
      state.currentSessionId = storedSessionId || null;
    } catch (e) {
      console.error('Lỗi khi đọc LocalStorage:', e);
      state.sessions = [];
      state.ocrVault = [];
    }
  }

  function saveSessions() {
    try {
      localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(state.sessions));
      if (state.currentSessionId) {
        localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION_ID, state.currentSessionId);
      }
    } catch (e) {
      console.warn('Lỗi khi lưu sessions vào LocalStorage:', e);
    }
  }

  function saveVault() {
    try {
      localStorage.setItem(STORAGE_KEYS.OCR_VAULT, JSON.stringify(state.ocrVault));
    } catch (e) {
      console.warn('Lỗi khi lưu vault vào LocalStorage:', e);
    }
  }

  function loadTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME) || 'dark-theme';
    state.theme = savedTheme;
    document.body.className = savedTheme;
  }

  function toggleTheme() {
    state.theme = state.theme === 'dark-theme' ? 'light-theme' : 'dark-theme';
    document.body.className = state.theme;
    localStorage.setItem(STORAGE_KEYS.THEME, state.theme);
  }

  // =========================================================================
  // SESSION LOGIC
  // =========================================================================
  function createNewSession(focusInput = true) {
    const newSession = {
      id: 'session_' + Date.now(),
      title: 'Cuộc trò chuyện mới',
      createdAt: Date.now(),
      messages: [],
    };

    state.sessions.unshift(newSession);
    state.currentSessionId = newSession.id;
    saveSessions();
    renderSidebarSessions();
    loadSession(newSession.id);

    if (focusInput && dom.promptInput) {
      dom.promptInput.focus();
    }
  }

  function getSessionById(id) {
    return state.sessions.find((s) => s.id === id);
  }

  function getCurrentSession() {
    return getSessionById(state.currentSessionId);
  }

  function loadSession(sessionId) {
    state.currentSessionId = sessionId;
    localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION_ID, sessionId);

    const session = getSessionById(sessionId);
    if (!session) return;

    if (dom.currentChatTitle) {
      dom.currentChatTitle.textContent = session.title || 'Cuộc trò chuyện mới';
    }

    renderMessages(session.messages);
    renderSidebarSessions();

    // Auto collapse sidebar on mobile when session selected
    if (window.innerWidth <= 768 && dom.sidebar) {
      dom.sidebar.classList.remove('open');
    }
  }

  function deleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    state.sessions = state.sessions.filter((s) => s.id !== sessionId);

    if (state.currentSessionId === sessionId) {
      if (state.sessions.length > 0) {
        loadSession(state.sessions[0].id);
      } else {
        createNewSession(false);
      }
    } else {
      saveSessions();
      renderSidebarSessions();
    }
  }

  function clearAllSessions() {
    if (confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử các cuộc trò chuyện?')) {
      state.sessions = [];
      localStorage.removeItem(STORAGE_KEYS.SESSIONS);
      createNewSession(false);
    }
  }

  // =========================================================================
  // BACKEND HEALTH CHECK
  // =========================================================================
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/v1/health', { method: 'GET' });
      if (res.ok && dom.systemStatus) {
        dom.systemStatus.innerHTML = '<span class="status-dot"></span><span class="status-label">Hệ thống Trực tuyến</span>';
      }
    } catch {
      if (dom.systemStatus) {
        dom.systemStatus.innerHTML = '<span class="status-dot" style="background-color: var(--accent-rose); box-shadow: 0 0 8px rgba(244,63,94,0.4);"></span><span class="status-label" style="color: var(--accent-rose);">Mất kết nối máy chủ</span>';
      }
    }
  }

  // =========================================================================
  // SIDEBAR RENDERING
  // =========================================================================
  function renderSidebarSessions(query = '') {
    if (!dom.sessionsToday || !dom.sessionsRecent) return;

    dom.sessionsToday.innerHTML = '';
    dom.sessionsRecent.innerHTML = '';

    const filter = query.trim().toLowerCase();
    const now = Date.now();
    const ONE_DAY_MS = 24 * 60 * 60 * 1000;

    let hasToday = false;
    let hasRecent = false;

    state.sessions.forEach((s) => {
      if (filter && !s.title.toLowerCase().includes(filter)) return;

      const isToday = now - s.createdAt < ONE_DAY_MS;
      const isActive = s.id === state.currentSessionId;

      const item = document.createElement('div');
      item.className = `session-item ${isActive ? 'active' : ''}`;
      item.innerHTML = `
        <svg class="session-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span class="session-item-title">${escapeHtml(s.title || 'Cuộc trò chuyện mới')}</span>
        <button class="session-item-del-btn" title="Xóa cuộc trò chuyện">✕</button>
      `;

      item.addEventListener('click', () => loadSession(s.id));
      const delBtn = item.querySelector('.session-item-del-btn');
      delBtn.addEventListener('click', (e) => deleteSession(s.id, e));

      if (isToday) {
        dom.sessionsToday.appendChild(item);
        hasToday = true;
      } else {
        dom.sessionsRecent.appendChild(item);
        hasRecent = true;
      }
    });

    if (!hasToday) {
      dom.sessionsToday.innerHTML = '<div class="vault-empty-hint">Chưa có lịch sử hôm nay.</div>';
    }
    if (!hasRecent) {
      dom.sessionsRecent.innerHTML = '<div class="vault-empty-hint">Chưa có lịch sử cũ hơn.</div>';
    }
  }

  function renderDocumentVault() {
    if (!dom.documentVaultList) return;
    dom.documentVaultList.innerHTML = '';

    if (!state.ocrVault || state.ocrVault.length === 0) {
      dom.documentVaultList.innerHTML = '<div class="vault-empty-hint">Chưa có tài liệu nào được bóc tách.</div>';
      return;
    }

    state.ocrVault.forEach((doc, idx) => {
      const item = document.createElement('div');
      item.className = 'vault-item';
      item.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        </svg>
        <span class="vault-item-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <span class="vault-item-badge">${doc.pageCount || 1} trang</span>
      `;
      item.addEventListener('click', () => openDocumentInDrawer(doc));
      dom.documentVaultList.appendChild(item);
    });
  }

  // =========================================================================
  // MESSAGE RENDERING & FORMATTING
  // =========================================================================
  function renderMessages(messages) {
    if (!dom.chatViewport || !dom.heroContainer || !dom.messagesContainer) return;

    if (!messages || messages.length === 0) {
      dom.heroContainer.style.display = 'flex';
      dom.messagesContainer.style.display = 'none';
      dom.messagesContainer.innerHTML = '';
      return;
    }

    dom.heroContainer.style.display = 'none';
    dom.messagesContainer.style.display = 'flex';
    dom.messagesContainer.innerHTML = '';

    messages.forEach((msg) => {
      appendMessageToDOM(msg, false);
    });

    scrollToBottom();
  }

  function appendMessageToDOM(msg, shouldScroll = true) {
    if (!dom.messagesContainer) return;

    dom.heroContainer.style.display = 'none';
    dom.messagesContainer.style.display = 'flex';

    const row = document.createElement('div');
    row.className = `message-row ${msg.role}`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (msg.role === 'user') {
      avatar.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
        </svg>
      `;
    } else {
      avatar.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>
        </svg>
      `;
    }

    // Content Wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';

    // Attached file preview for user
    if (msg.role === 'user' && msg.attachedFile) {
      const fileBadge = document.createElement('div');
      fileBadge.className = 'attached-file-badge';
      fileBadge.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        </svg>
        <span>${escapeHtml(msg.attachedFile.name)} (${formatFileSize(msg.attachedFile.size)})</span>
      `;
      wrapper.appendChild(fileBadge);
    }

    // Thinking process accordion for assistant
    if (msg.role === 'assistant' && msg.thinkingSteps && msg.thinkingSteps.length > 0) {
      const accordion = createThinkingAccordionElement(msg.thinkingSteps, false);
      wrapper.appendChild(accordion);
    }

    // Bubble
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (msg.role === 'user') {
      bubble.textContent = msg.content;
    } else {
      bubble.innerHTML = renderMarkdown(msg.content);
    }
    wrapper.appendChild(bubble);

    // Citations Chips
    if (msg.citations && msg.citations.length > 0) {
      const citBox = document.createElement('div');
      citBox.style.marginTop = '10px';
      msg.citations.forEach((c) => {
        const chip = document.createElement('span');
        chip.className = 'citation-chip';
        chip.textContent = c;
        citBox.appendChild(chip);
      });
      wrapper.appendChild(citBox);
    }

    row.appendChild(avatar);
    row.appendChild(wrapper);
    dom.messagesContainer.appendChild(row);

    if (shouldScroll) {
      scrollToBottom();
    }

    return { row, wrapper, bubble };
  }

  function createThinkingAccordionElement(steps, isLive = false) {
    const acc = document.createElement('div');
    acc.className = 'thinking-accordion';

    const header = document.createElement('div');
    header.className = 'thinking-header';
    header.innerHTML = `
      <div class="thinking-title">
        ${isLive ? '<div class="thinking-spinner"></div>' : '<span>🧠</span>'}
        <span class="thinking-title-text">${isLive ? 'Quy trình tư duy tác tử (LangGraph)...' : `Đã hoàn thành phân tích (${steps.length} bước)`}</span>
      </div>
      <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    `;

    const stepsContainer = document.createElement('div');
    stepsContainer.className = 'thinking-steps';

    steps.forEach((st) => {
      const stepItem = document.createElement('div');
      stepItem.className = 'thinking-step-item';
      stepItem.innerHTML = `
        <span class="step-icon ${st.status === 'completed' ? 'done' : 'running'}">
          ${st.status === 'completed' ? '✓' : '●'}
        </span>
        <span class="step-text">${escapeHtml(st.message)}</span>
      `;
      stepsContainer.appendChild(stepItem);
    });

    header.addEventListener('click', () => {
      acc.classList.toggle('open');
    });

    acc.appendChild(header);
    acc.appendChild(stepsContainer);
    return acc;
  }

  // =========================================================================
  // MARKDOWN & SYNTAX RENDERING
  // =========================================================================
  function renderMarkdown(rawText) {
    if (!rawText) return '';

    let text = rawText;

    // Redline Diff block syntax detection
    text = text.replace(/```diff([\s\S]*?)```/g, (match, diffContent) => {
      const lines = diffContent.trim().split('\n');
      const formattedLines = lines.map((line) => {
        if (line.startsWith('-')) {
          return `<div class="diff-line deletion">${escapeHtml(line)}</div>`;
        } else if (line.startsWith('+')) {
          return `<div class="diff-line addition">${escapeHtml(line)}</div>`;
        }
        return `<div class="diff-line">${escapeHtml(line)}</div>`;
      });
      return `<div class="diff-container">${formattedLines.join('')}</div>`;
    });

    // Standard Code Blocks
    text = text.replace(/```([a-zA-Z0-9_-]*)([\s\S]*?)```/g, (match, lang, code) => {
      return `<div class="code-block-wrapper">
        <div class="code-header">
          <span>${lang || 'code'}</span>
          <button class="copy-code-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='Đã chép!'; setTimeout(()=>this.textContent='Sao chép', 2000);">Sao chép</button>
        </div>
        <pre><code>${escapeHtml(code.trim())}</code></pre>
      </div>`;
    });

    // Inline Code
    text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Headings
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italic
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Markdown Tables
    text = text.replace(/(\|[^\n]+\|\r?\n)((?:\|[^\n]+\|\r?\n?)+)/g, (match) => {
      const rows = match.trim().split('\n');
      if (rows.length < 2) return match;

      let html = '<div class="table-container"><table class="markdown-table">';
      let isHeader = true;

      rows.forEach((row, i) => {
        // Skip markdown separator row |---|---|
        if (row.includes('---')) {
          isHeader = false;
          return;
        }

        const cols = row.split('|').filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
        if (isHeader) {
          html += '<thead><tr>';
          cols.forEach((col) => (html += `<th>${col.trim()}</th>`));
          html += '</tr></thead><tbody>';
          isHeader = false;
        } else {
          html += '<tr>';
          cols.forEach((col) => (html += `<td>${col.trim()}</td>`));
          html += '</tr>';
        }
      });

      html += '</tbody></table></div>';
      return html;
    });

    // Bullet Lists
    text = text.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
    text = text.replace(/(<li>[\s\S]*?<\/li>)/gim, '<ul class="markdown-list">$1</ul>');

    // Linebreaks to paragraphs
    text = text
      .split('\n\n')
      .map((p) => {
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<h') || p.startsWith('<div') || p.startsWith('<pre') || p.startsWith('<ul') || p.startsWith('<table')) {
          return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
      })
      .join('');

    // Citations highlighting [Điều 468 BLDS 2015]
    text = text.replace(/\[((?:Điều|Khoản|Bộ luật|Luật|Án lệ|Nghị định)[^\]]+)\]/g, '<span class="citation-chip">[$1]</span>');

    return text;
  }

  // =========================================================================
  // SEND MESSAGE & SSE STREAM CONTROLLER
  // =========================================================================
  async function sendMessage() {
    const text = dom.promptInput.value.trim();
    const file = state.attachedFile;

    if (!text && !file) return;
    if (state.isGenerating) return;

    // Ensure session exists
    let session = getCurrentSession();
    if (!session) {
      createNewSession(false);
      session = getCurrentSession();
    }

    // Auto title session from first user message
    if (session.messages.length === 0) {
      const cleanTitle = (text || file.name).slice(0, 42).replace(/[\r\n]+/g, ' ');
      session.title = cleanTitle;
      if (dom.currentChatTitle) dom.currentChatTitle.textContent = cleanTitle;
    }

    // Push User Message
    const userMsg = {
      role: 'user',
      content: text,
      attachedFile: file ? { name: file.name, size: file.size, type: file.type } : null,
      timestamp: Date.now(),
    };
    session.messages.push(userMsg);
    saveSessions();
    renderSidebarSessions();

    appendMessageToDOM(userMsg, true);

    // Clear prompt bar
    dom.promptInput.value = '';
    dom.promptInput.style.height = 'auto';
    clearAttachment();

    // Setup Assistant Streaming Placeholder
    setGeneratingState(true);

    const thinkingSteps = [];
    const assistantMsg = {
      role: 'assistant',
      content: '',
      thinkingSteps: thinkingSteps,
      citations: [],
      timestamp: Date.now(),
    };

    const domRef = appendMessageToDOM(assistantMsg, true);
    const bubbleEl = domRef.bubble;
    const wrapperEl = domRef.wrapper;

    // Create live thinking accordion
    const liveAccordion = createThinkingAccordionElement(thinkingSteps, true);
    liveAccordion.classList.add('open');
    wrapperEl.insertBefore(liveAccordion, bubbleEl);

    // Prepare FormData
    const formData = new FormData();
    if (text) formData.append('message', text);
    if (session.id) formData.append('session_id', session.id);
    if (file) formData.append('file', file);

    state.activeController = new AbortController();

    try {
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        body: formData,
        signal: state.activeController.signal,
      });

      if (!response.ok) {
        throw new Error(`Máy chủ phản hồi mã lỗi: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop(); // keep last incomplete event

        for (const evtStr of events) {
          const line = evtStr.trim();
          if (!line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'step') {
              // Update live thinking steps
              const stepInfo = STEP_METADATA[data.step] || { label: data.step, icon: '⚡' };
              const stepMsg = data.message || stepInfo.label;

              thinkingSteps.push({
                step: data.step,
                status: data.status || 'running',
                message: stepMsg,
              });

              // Re-render accordion content
              const stepsCont = liveAccordion.querySelector('.thinking-steps');
              if (stepsCont) {
                const item = document.createElement('div');
                item.className = 'thinking-step-item';
                item.innerHTML = `
                  <span class="step-icon ${data.status === 'completed' ? 'done' : 'running'}">
                    ${data.status === 'completed' ? '✓' : '●'}
                  </span>
                  <span class="step-text">${escapeHtml(stepMsg)}</span>
                `;
                stepsCont.appendChild(item);
              }
              scrollToBottom();

            } else if (data.type === 'chat_completed') {
              assistantMsg.content = data.answer || '';
              assistantMsg.citations = data.citations || [];
              bubbleEl.innerHTML = renderMarkdown(assistantMsg.content);

              if (data.citations && data.citations.length > 0) {
                const citBox = document.createElement('div');
                citBox.style.marginTop = '12px';
                data.citations.forEach((c) => {
                  const chip = document.createElement('span');
                  chip.className = 'citation-chip';
                  chip.textContent = c;
                  citBox.appendChild(chip);
                });
                wrapperEl.appendChild(citBox);
              }

            } else if (data.type === 'audit_completed') {
              const rep = data.report;
              if (rep) {
                renderAuditReportIntoBubble(rep, bubbleEl, wrapperEl);
                assistantMsg.content = bubbleEl.innerText;
              } else {
                bubbleEl.innerHTML = `<p>${data.message || 'Thẩm định hợp đồng hoàn tất.'}</p>`;
              }

            } else if (data.type === 'error') {
              bubbleEl.innerHTML = `<div style="color: var(--accent-rose); font-weight: 500;">❌ ${escapeHtml(data.message)}</div>`;
            }
          } catch (e) {
            console.warn('Lỗi phân tích cú pháp SSE:', e, line);
          }
        }
      }

      // Finish thinking accordion state
      const titleText = liveAccordion.querySelector('.thinking-title-text');
      const spinner = liveAccordion.querySelector('.thinking-spinner');
      if (spinner) spinner.remove();
      if (titleText) {
        titleText.textContent = `Đã hoàn thành phân tích (${thinkingSteps.length} bước)`;
      }
      liveAccordion.classList.remove('open');

      session.messages.push(assistantMsg);
      saveSessions();
    } catch (err) {
      if (err.name === 'AbortError') {
        bubbleEl.innerHTML += '<p style="color: var(--text-muted); font-style: italic;">[Đã dừng tạo phản hồi]</p>';
      } else {
        bubbleEl.innerHTML = `<div style="color: var(--accent-rose);">⚠️ Đã xảy ra lỗi kết nối: ${escapeHtml(err.message)}</div>`;
      }
    } finally {
      setGeneratingState(false);
      state.activeController = null;
      scrollToBottom();
    }
  }

  function renderAuditReportIntoBubble(rep, bubbleEl, wrapperEl) {
    let html = '';

    // Overall Health Score Card
    const score = rep.overall_score || 85;
    const risk = rep.risk_level || 'Trung bình';
    const issuesCount = (rep.vulnerabilities || []).length;
    const scoreClass = score >= 80 ? 'green' : score >= 60 ? 'yellow' : 'red';

    html += `
      <div class="health-score-card">
        <div class="score-circle ${scoreClass}">
          <span class="score-num">${score}</span>
          <span class="score-max">/100</span>
        </div>
        <div class="score-meta">
          <div class="score-title">Điểm Sức khỏe Pháp lý: ${score}/100 (${risk})</div>
          <div class="score-desc">Hệ thống phát hiện ${issuesCount} điểm rủi ro và các bẫy điều khoản cần xem xét sửa đổi.</div>
        </div>
      </div>
    `;

    // Executive Summary
    if (rep.executive_summary) {
      html += `<h3>Tóm tắt Đánh giá Pháp lý</h3><p>${escapeHtml(rep.executive_summary)}</p>`;
    }

    // Vulnerabilities
    if (rep.vulnerabilities && rep.vulnerabilities.length > 0) {
      html += `<h3>Chi tiết Lỗ hổng & Bẫy Điều khoản</h3>`;
      rep.vulnerabilities.forEach((v, idx) => {
        const sevClass = v.severity === 'HIGH' ? 'red' : v.severity === 'MEDIUM' ? 'yellow' : 'green';
        html += `
          <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 12px 16px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <strong>#${idx + 1}. ${escapeHtml(v.clause_id || 'Điều khoản')}</strong>
              <span class="score-circle ${sevClass}" style="width: auto; height: auto; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem;">Mức độ: ${v.severity}</span>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 6px;"><strong>Mô tả:</strong> ${escapeHtml(v.issue_description)}</p>
            ${v.legal_ground ? `<p style="font-size: 0.8rem; color: var(--text-code);"><strong>Căn cứ:</strong> ${escapeHtml(v.legal_ground)}</p>` : ''}
          </div>
        `;
      });
    }

    // Redline Recommendations
    if (rep.redline_recommendations && rep.redline_recommendations.length > 0) {
      html += `<h3>Đề xuất Chỉnh lý Điều khoản (Redline Diffs)</h3>`;
      rep.redline_recommendations.forEach((r) => {
        html += `
          <div style="margin: 12px 0;">
            <div style="font-weight: 600; font-size: 0.88rem; color: var(--accent-cyan); margin-bottom: 4px;">${escapeHtml(r.clause_id || 'Đề xuất sửa đổi')}</div>
            <div class="diff-container">
              <div class="diff-line deletion">- ${escapeHtml(r.original_text || '')}</div>
              <div class="diff-line addition">+ ${escapeHtml(r.suggested_revision || '')}</div>
            </div>
            ${r.rationale ? `<p style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;"><strong>Lý do sửa đổi:</strong> ${escapeHtml(r.rationale)}</p>` : ''}
          </div>
        `;
      });
    }

    bubbleEl.innerHTML = html;
  }

  function stopGenerating() {
    if (state.activeController) {
      state.activeController.abort();
    }
  }

  function setGeneratingState(isGen) {
    state.isGenerating = isGen;
    if (dom.sendIcon && dom.stopIcon) {
      if (isGen) {
        dom.sendIcon.style.display = 'none';
        dom.stopIcon.style.display = 'block';
        dom.sendBtn.title = 'Dừng tạo câu trả lời';
      } else {
        dom.sendIcon.style.display = 'block';
        dom.stopIcon.style.display = 'none';
        dom.sendBtn.title = 'Gửi (Enter)';
      }
    }
  }

  // =========================================================================
  // ATTACHMENT & OCR DRAWER
  // =========================================================================
  function setAttachment(file) {
    state.attachedFile = file;
    if (!dom.attachmentsPreviewBar) return;

    dom.attachmentsPreviewBar.innerHTML = `
      <div class="attachment-pill">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 14px; height: 14px; color: var(--accent-cyan);">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        </svg>
        <span>${escapeHtml(file.name)} (${formatFileSize(file.size)})</span>
        <button class="attachment-remove-btn" id="remove-attachment-btn" title="Xóa đính kèm">✕</button>
      </div>
    `;
    dom.attachmentsPreviewBar.style.display = 'flex';

    const rmBtn = document.getElementById('remove-attachment-btn');
    if (rmBtn) {
      rmBtn.addEventListener('click', clearAttachment);
    }
  }

  function clearAttachment() {
    state.attachedFile = null;
    if (dom.fileUploadInput) dom.fileUploadInput.value = '';
    if (dom.attachmentsPreviewBar) {
      dom.attachmentsPreviewBar.innerHTML = '';
      dom.attachmentsPreviewBar.style.display = 'none';
    }
  }

  async function handleOcrUpload(file) {
    if (!file) return;

    openDrawer();
    if (dom.drawerDocName) dom.drawerDocName.textContent = file.name;
    if (dom.drawerPageCount) dom.drawerPageCount.textContent = '...';
    if (dom.drawerTableCount) dom.drawerTableCount.textContent = '...';
    if (dom.drawerCharCount) dom.drawerCharCount.textContent = '...';
    if (dom.drawerMarkdownPreview) {
      dom.drawerMarkdownPreview.innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; color: var(--text-muted); gap: 12px;">
          <div class="thinking-spinner" style="width: 24px; height: 24px; border-width: 3px;"></div>
          <span>Đang bóc tách layout, OCR văn bản và cấu trúc bảng biểu...</span>
        </div>
      `;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/ocr', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`OCR thất bại: mã phản hồi ${res.status}`);
      }

      const data = await res.json();
      if (dom.drawerPageCount) dom.drawerPageCount.textContent = data.page_count || 1;
      if (dom.drawerTableCount) dom.drawerTableCount.textContent = (data.tables || []).length;
      if (dom.drawerCharCount) dom.drawerCharCount.textContent = data.char_count || 0;
      if (dom.drawerMarkdownPreview) {
        dom.drawerMarkdownPreview.textContent = data.markdown || '';
      }

      // Save to Vault
      const vaultEntry = {
        filename: data.filename || file.name,
        pageCount: data.page_count || 1,
        tableCount: (data.tables || []).length,
        charCount: data.char_count || 0,
        markdown: data.markdown || '',
        timestamp: Date.now(),
      };
      state.ocrVault.unshift(vaultEntry);
      saveVault();
      renderDocumentVault();
    } catch (err) {
      if (dom.drawerMarkdownPreview) {
        dom.drawerMarkdownPreview.innerHTML = `<div style="color: var(--accent-rose); padding: 20px;">❌ ${escapeHtml(err.message)}</div>`;
      }
    }
  }

  function openDrawer() {
    if (dom.drawerOverlay && dom.documentDrawer) {
      dom.drawerOverlay.classList.add('open');
      dom.documentDrawer.classList.add('open');
    }
  }

  function closeDrawer() {
    if (dom.drawerOverlay && dom.documentDrawer) {
      dom.drawerOverlay.classList.remove('open');
      dom.documentDrawer.classList.remove('open');
    }
  }

  function openDocumentInDrawer(doc) {
    openDrawer();
    if (dom.drawerDocName) dom.drawerDocName.textContent = doc.filename;
    if (dom.drawerPageCount) dom.drawerPageCount.textContent = doc.pageCount;
    if (dom.drawerTableCount) dom.drawerTableCount.textContent = doc.tableCount;
    if (dom.drawerCharCount) dom.drawerCharCount.textContent = doc.charCount;
    if (dom.drawerMarkdownPreview) {
      dom.drawerMarkdownPreview.textContent = doc.markdown;
    }
  }

  // =========================================================================
  // EXPORT CHAT TO MARKDOWN
  // =========================================================================
  function exportChatToMarkdown() {
    const session = getCurrentSession();
    if (!session || !session.messages || session.messages.length === 0) {
      alert('Chưa có nội dung cuộc trò chuyện để xuất.');
      return;
    }

    let md = `# LexiGuard AI — Báo Cáo Hội Thoại Pháp Lý\n`;
    md += `**Chủ đề:** ${session.title}\n`;
    md += `**Thời gian tạo:** ${new Date(session.createdAt).toLocaleString('vi-VN')}\n\n`;
    md += `---\n\n`;

    session.messages.forEach((m, idx) => {
      const roleName = m.role === 'user' ? '👤 NGƯỜI DÙNG' : '⚖️ LEXIGUARD AI';
      md += `### ${roleName}\n\n`;
      if (m.attachedFile) {
        md += `> Đính kèm tệp: **${m.attachedFile.name}**\n\n`;
      }
      md += `${m.content}\n\n`;
      if (m.citations && m.citations.length > 0) {
        md += `**Căn cứ pháp luật:** ${m.citations.join(' | ')}\n\n`;
      }
      md += `---\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LexiGuard_${(session.title || 'chat').replace(/[^a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // =========================================================================
  // UTILITIES & EVENT BINDINGS
  // =========================================================================
  function scrollToBottom() {
    if (dom.chatViewport) {
      dom.chatViewport.scrollTo({
        top: dom.chatViewport.scrollHeight,
        behavior: 'smooth',
      });
    }
  }

  function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function bindEvents() {
    // Send / Stop
    if (dom.sendBtn) {
      dom.sendBtn.addEventListener('click', () => {
        if (state.isGenerating) {
          stopGenerating();
        } else {
          sendMessage();
        }
      });
    }

    // Auto-expand textarea & Enter to send
    if (dom.promptInput) {
      dom.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!state.isGenerating) sendMessage();
        }
      });

      dom.promptInput.addEventListener('input', () => {
        dom.promptInput.style.height = 'auto';
        dom.promptInput.style.height = Math.min(dom.promptInput.scrollHeight, 180) + 'px';
      });
    }

    // New Chat
    if (dom.newChatBtn) {
      dom.newChatBtn.addEventListener('click', () => createNewSession(true));
    }

    // Keyboard Shortcut Ctrl+K
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        createNewSession(true);
      }
    });

    // Sidebar Toggles
    if (dom.sidebarOpenBtn && dom.sidebar) {
      dom.sidebarOpenBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          dom.sidebar.classList.add('open');
        } else {
          dom.sidebar.classList.remove('collapsed');
        }
      });
    }

    if (dom.sidebarCloseBtn && dom.sidebar) {
      dom.sidebarCloseBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          dom.sidebar.classList.remove('open');
        } else {
          dom.sidebar.classList.add('collapsed');
        }
      });
    }

    // Theme Toggle
    if (dom.themeToggleBtn) {
      dom.themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // Clear All Sessions
    if (dom.clearAllSessionsBtn) {
      dom.clearAllSessionsBtn.addEventListener('click', clearAllSessions);
    }

    // Export Chat
    if (dom.exportChatBtn) {
      dom.exportChatBtn.addEventListener('click', exportChatToMarkdown);
    }

    // Search Sessions
    if (dom.sessionSearchInput) {
      dom.sessionSearchInput.addEventListener('input', (e) => {
        const val = e.target.value;
        if (dom.clearSearchBtn) dom.clearSearchBtn.style.display = val ? 'block' : 'none';
        renderSidebarSessions(val);
      });
    }
    if (dom.clearSearchBtn && dom.sessionSearchInput) {
      dom.clearSearchBtn.addEventListener('click', () => {
        dom.sessionSearchInput.value = '';
        dom.clearSearchBtn.style.display = 'none';
        renderSidebarSessions('');
      });
    }

    // Hero Prompt Cards
    dom.promptCards.forEach((card) => {
      card.addEventListener('click', () => {
        if (card.id === 'trigger-ocr-card') {
          if (dom.ocrUploadInput) dom.ocrUploadInput.click();
          return;
        }
        const promptText = card.getAttribute('data-prompt');
        if (promptText && dom.promptInput) {
          dom.promptInput.value = promptText;
          dom.promptInput.style.height = 'auto';
          dom.promptInput.style.height = Math.min(dom.promptInput.scrollHeight, 180) + 'px';
          sendMessage();
        }
      });
    });

    // File Attachment
    if (dom.attachFileBtn && dom.fileUploadInput) {
      dom.attachFileBtn.addEventListener('click', () => dom.fileUploadInput.click());
      dom.fileUploadInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          setAttachment(e.target.files[0]);
        }
      });
    }

    // OCR Upload & Drawer
    if (dom.ocrBtn && dom.ocrUploadInput) {
      dom.ocrBtn.addEventListener('click', () => dom.ocrUploadInput.click());
      dom.ocrUploadInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleOcrUpload(e.target.files[0]);
        }
      });
    }

    if (dom.openDrawerBtn) {
      dom.openDrawerBtn.addEventListener('click', openDrawer);
    }
    if (dom.drawerCloseBtn) {
      dom.drawerCloseBtn.addEventListener('click', closeDrawer);
    }
    if (dom.drawerOverlay) {
      dom.drawerOverlay.addEventListener('click', closeDrawer);
    }

    // Drawer Actions
    if (dom.copyOcrTextBtn && dom.drawerMarkdownPreview) {
      dom.copyOcrTextBtn.addEventListener('click', () => {
        const text = dom.drawerMarkdownPreview.innerText;
        if (!text || text.includes('Chưa có văn bản')) return;
        navigator.clipboard.writeText(text);
        dom.copyOcrTextBtn.textContent = 'Đã chép vào clipboard!';
        setTimeout(() => (dom.copyOcrTextBtn.textContent = 'Sao chép văn bản'), 2000);
      });
    }

    if (dom.sendOcrToChatBtn && dom.drawerMarkdownPreview && dom.promptInput) {
      dom.sendOcrToChatBtn.addEventListener('click', () => {
        const text = dom.drawerMarkdownPreview.innerText;
        if (!text || text.includes('Chưa có văn bản')) return;
        closeDrawer();
        dom.promptInput.value = `Thẩm định các điều khoản và bảng biểu sau đây được bóc tách từ tài liệu OCR:\n\n${text}`;
        dom.promptInput.focus();
        dom.promptInput.style.height = 'auto';
        dom.promptInput.style.height = Math.min(dom.promptInput.scrollHeight, 180) + 'px';
      });
    }

    // Drag and Drop Files onto Prompt Box & Chat Viewport
    const dropZone = document.querySelector('.chat-canvas');
    if (dropZone) {
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
        });
      });

      dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
          const file = files[0];
          const ext = file.name.split('.').pop().toLowerCase();
          if (['png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp'].includes(ext)) {
            handleOcrUpload(file);
          } else {
            setAttachment(file);
          }
        }
      });
    }
  }

  // =========================================================================
  // BOOTSTRAP ON LOAD
  // =========================================================================
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
