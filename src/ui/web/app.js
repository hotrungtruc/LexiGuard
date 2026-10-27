/**
 * LexiGuard AI — Web Application Controller (Gemini & ChatGPT Inspired)
 * Multi-Agent Legal Assistant & Contract Auditor
 * Minimalist, User-Friendly & Multilingual (VI / EN)
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
    LANG: 'lexiguard_lang',
  };

  // =========================================================================
  // MULTILINGUAL DICTIONARY (I18N)
  // =========================================================================
  const I18N = {
    vi: {
      documentTitle: 'LexiGuard AI — Trợ lý Pháp lý',
      newChat: 'Cuộc trò chuyện mới',
      searchPlaceholder: 'Tìm kiếm lịch sử...',
      today: 'Hôm nay',
      recent: '7 ngày trước',
      vaultTitle: 'Thư viện OCR',
      vaultEmpty: 'Chưa có tài liệu nào.',
      todayEmpty: 'Chưa có lịch sử hôm nay.',
      recentEmpty: 'Chưa có lịch sử cũ hơn.',
      systemOnline: 'Trực tuyến',
      systemOffline: 'Mất kết nối',
      chatTitleDefault: 'Cuộc trò chuyện mới',
      ocrBtn: 'Tài liệu OCR',
      exportBtn: 'Xuất',
      heroGreeting: 'Hôm nay tôi có thể giúp gì cho bạn?',
      promptPlaceholder: 'Hỏi bất cứ điều gì hoặc đính kèm tài liệu...',
      disclaimer: 'LexiGuard có thể mắc lỗi. Vui lòng kiểm tra các điều khoản quan trọng.',
      chipAudit: 'Thẩm định hợp đồng',
      chipAuditPrompt: 'Thẩm định giúp tôi các rủi ro pháp lý phổ biến trong hợp đồng dịch vụ tư vấn và đề xuất điều khoản bảo vệ bên thuê.',
      chipLookup: 'Tra cứu luật định',
      chipLookupPrompt: 'Theo Bộ luật Dân sự 2015, mức lãi suất vay tối đa mà các bên được thỏa thuận là bao nhiêu? Hậu quả pháp lý nếu vượt trần?',
      chipTable: 'Kiểm toán bảng tính',
      chipTablePrompt: 'Kiểm tra đối soát bảng phụ lục chi phí:\n| STT | Dịch vụ | SL | Đơn giá | Thành tiền |\n| 1 | Thẩm định | 2 | 10.000.000 | 25.000.000 |\n| Tổng | | | | 25.000.000 |',
      chipOcr: 'Bóc tách OCR tài liệu',
      attachFileTitle: 'Đính kèm tệp (.pdf, .docx, .md)',
      ocrBtnTitle: 'Bóc tách OCR từ ảnh hoặc scan',
      sendBtnTitle: 'Gửi (Enter)',
      stopBtnTitle: 'Dừng tạo câu trả lời',
      clearAllConfirm: 'Bạn có chắc chắn muốn xóa toàn bộ lịch sử?',
      drawerTitle: 'Tài liệu OCR & Bảng biểu',
      drawerEmpty: 'Chưa có văn bản OCR nào. Tải ảnh hoặc PDF scan để bắt đầu.',
      drawerPages: 'Trang',
      drawerTables: 'Bảng',
      drawerChars: 'Ký tự',
      drawerTextTitle: 'Nội dung văn bản bóc tách:',
      drawerCopyBtn: 'Sao chép văn bản',
      drawerSendBtn: 'Đưa vào kiểm toán',
      drawerCopied: 'Đã sao chép vào clipboard!',
      themeLightDark: 'Chuyển giao diện Sáng / Tối',
      clearAllTooltip: 'Xóa toàn bộ lịch sử',
      healthScoreTitle: 'Điểm Sức khỏe Pháp lý',
      vulnerabilitiesTitle: 'Chi tiết Lỗ hổng & Bẫy Điều khoản',
      redlineTitle: 'Đề xuất Chỉnh lý Điều khoản (Redline)',
      copyCode: 'Sao chép',
      copiedCode: 'Đã chép!',
      levelRisk: 'Rủi ro',
      exportHeader: '# LexiGuard AI — Báo Cáo Hội Thoại Pháp Lý',
      exportEmpty: 'Chưa có nội dung cuộc trò chuyện để xuất.',
      userLabel: '👤 NGƯỜI DÙNG',
      assistantLabel: '⚖️ LEXIGUARD AI',
      attachedFileLabel: 'Đính kèm tệp',
      legalGroundsLabel: 'Căn cứ pháp luật',
      ocrPromptPrefix: 'Thẩm định các điều khoản và bảng biểu sau đây được bóc tách từ tài liệu OCR:\n\n',
      thinkingProcessRunning: 'Quy trình tư duy tác tử...',
      thinkingProcessDone: 'Đã hoàn thành phân tích',
    },
    en: {
      documentTitle: 'LexiGuard AI — Legal Assistant',
      newChat: 'New chat',
      searchPlaceholder: 'Search history...',
      today: 'Today',
      recent: 'Previous 7 days',
      vaultTitle: 'OCR Library',
      vaultEmpty: 'No documents yet.',
      todayEmpty: 'No conversations today.',
      recentEmpty: 'No previous conversations.',
      systemOnline: 'Online',
      systemOffline: 'Offline',
      chatTitleDefault: 'New chat',
      ocrBtn: 'OCR Docs',
      exportBtn: 'Export',
      heroGreeting: 'How can I help you today?',
      promptPlaceholder: 'Ask anything or attach a contract...',
      disclaimer: 'LexiGuard can make mistakes. Please verify important legal clauses.',
      chipAudit: 'Audit Contract',
      chipAuditPrompt: 'Review the common legal risks in a consulting service agreement and propose protective clauses for the client.',
      chipLookup: 'Legal Lookup',
      chipLookupPrompt: 'Under the Civil Code 2015, what is the maximum agreed interest rate? What are the legal consequences of exceeding the ceiling?',
      chipTable: 'Audit Calculations',
      chipTablePrompt: 'Verify the arithmetic of this cost schedule:\n| No | Service | Qty | Unit Price | Total |\n| 1 | Legal Audit | 2 | 10,000,000 | 25,000,000 |\n| Total | | | | 25,000,000 |',
      chipOcr: 'OCR Document Extract',
      attachFileTitle: 'Attach document (.pdf, .docx, .md)',
      ocrBtnTitle: 'Extract text/tables from scan or photo',
      sendBtnTitle: 'Send (Enter)',
      stopBtnTitle: 'Stop generating',
      clearAllConfirm: 'Are you sure you want to clear all history?',
      drawerTitle: 'OCR Document & Tables',
      drawerEmpty: 'No OCR document yet. Upload an image or scanned PDF to start.',
      drawerPages: 'Pages',
      drawerTables: 'Tables',
      drawerChars: 'Characters',
      drawerTextTitle: 'Extracted text content:',
      drawerCopyBtn: 'Copy text',
      drawerSendBtn: 'Send to audit',
      drawerCopied: 'Copied to clipboard!',
      themeLightDark: 'Toggle Light / Dark mode',
      clearAllTooltip: 'Clear all history',
      healthScoreTitle: 'Legal Health Score',
      vulnerabilitiesTitle: 'Identified Vulnerabilities & Traps',
      redlineTitle: 'Proposed Contract Revisions (Redline)',
      copyCode: 'Copy',
      copiedCode: 'Copied!',
      levelRisk: 'Risk',
      exportHeader: '# LexiGuard AI — Legal Conversation Report',
      exportEmpty: 'No conversation content to export.',
      userLabel: '👤 USER',
      assistantLabel: '⚖️ LEXIGUARD AI',
      attachedFileLabel: 'Attached file',
      legalGroundsLabel: 'Legal grounds',
      ocrPromptPrefix: 'Audit the following clauses and financial tables extracted from OCR:\n\n',
      thinkingProcessRunning: 'Agent reasoning process...',
      thinkingProcessDone: 'Analysis completed',
    },
  };

  const STEP_METADATA = {
    vi: {
      init: { label: 'Khởi tạo quy trình thẩm định...', icon: '⚙️' },
      extract_clauses: { label: 'Trích xuất cấu trúc điều khoản & bảng biểu', icon: '📑' },
      retrieve_statutes: { label: 'Truy vấn đồ thị tri thức pháp lý (GraphRAG)', icon: '⚖️' },
      audit_vulnerabilities: { label: 'Phân tích lỗ hổng & đề xuất Redline', icon: '🔍' },
      grade_hallucination: { label: 'Đối soát luật định & kiểm tra ảo giác', icon: '🛡️' },
      search: { label: 'Tra cứu cơ sở tri thức luật', icon: '🔎' },
      retrieved: { label: 'Kết nối văn bản & án lệ liên quan', icon: '📚' },
      synthesizing: { label: 'Tổng hợp câu trả lời...', icon: '✍️' },
    },
    en: {
      init: { label: 'Initializing audit pipeline...', icon: '⚙️' },
      extract_clauses: { label: 'Extracting clauses & financial tables', icon: '📑' },
      retrieve_statutes: { label: 'Querying legal knowledge graph (GraphRAG)', icon: '⚖️' },
      audit_vulnerabilities: { label: 'Auditing vulnerabilities & drafting Redline', icon: '🔍' },
      grade_hallucination: { label: 'Validating citations & hallucination check', icon: '🛡️' },
      search: { label: 'Searching legal knowledge base', icon: '🔎' },
      retrieved: { label: 'Connected relevant statutes & precedents', icon: '📚' },
      synthesizing: { label: 'Synthesizing response...', icon: '✍️' },
    },
  };

  // =========================================================================
  // APPLICATION STATE
  // =========================================================================
  let state = {
    lang: 'vi',
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
    langToggleBtn: document.getElementById('lang-toggle-btn'),
    langViCode: document.getElementById('lang-vi-code'),
    langEnCode: document.getElementById('lang-en-code'),
    chatViewport: document.getElementById('chat-viewport'),
    heroContainer: document.getElementById('hero-container'),
    messagesContainer: document.getElementById('messages-container'),
    promptChips: document.querySelectorAll('.prompt-chip'),
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
    loadLanguage();
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
  // LANGUAGE & I18N
  // =========================================================================
  function t(key) {
    const dict = I18N[state.lang] || I18N.vi;
    return dict[key] || key;
  }

  function loadLanguage() {
    const savedLang = localStorage.getItem(STORAGE_KEYS.LANG) || 'vi';
    setLanguage(savedLang, false);
  }

  function setLanguage(lang, shouldSave = true) {
    state.lang = lang === 'en' ? 'en' : 'vi';
    if (shouldSave) {
      localStorage.setItem(STORAGE_KEYS.LANG, state.lang);
    }

    document.documentElement.lang = state.lang;
    document.title = t('documentTitle');

    // Update Language Toggle Indicator
    if (dom.langViCode && dom.langEnCode) {
      if (state.lang === 'vi') {
        dom.langViCode.classList.add('active');
        dom.langEnCode.classList.remove('active');
      } else {
        dom.langViCode.classList.remove('active');
        dom.langEnCode.classList.add('active');
      }
    }

    // Update all elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      if (key && I18N[state.lang][key]) {
        el.textContent = I18N[state.lang][key];
      }
    });

    // Update Placeholders & Tooltips
    if (dom.promptInput) {
      dom.promptInput.placeholder = t('promptPlaceholder');
    }
    if (dom.sessionSearchInput) {
      dom.sessionSearchInput.placeholder = t('searchPlaceholder');
    }
    if (dom.attachFileBtn) dom.attachFileBtn.title = t('attachFileTitle');
    if (dom.ocrBtn) dom.ocrBtn.title = t('ocrBtnTitle');
    if (dom.sendBtn && !state.isGenerating) dom.sendBtn.title = t('sendBtnTitle');
    if (dom.themeToggleBtn) dom.themeToggleBtn.title = t('themeLightDark');
    if (dom.clearAllSessionsBtn) dom.clearAllSessionsBtn.title = t('clearAllTooltip');

    // Update default session title if untouched
    const currentSession = getCurrentSession();
    if (currentSession && (currentSession.title === I18N.vi.chatTitleDefault || currentSession.title === I18N.en.chatTitleDefault)) {
      currentSession.title = t('chatTitleDefault');
      if (dom.currentChatTitle) dom.currentChatTitle.textContent = currentSession.title;
    }

    renderSidebarSessions();
  }

  function toggleLanguage() {
    const nextLang = state.lang === 'vi' ? 'en' : 'vi';
    setLanguage(nextLang, true);
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
      title: t('chatTitleDefault'),
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
      dom.currentChatTitle.textContent = session.title || t('chatTitleDefault');
    }

    renderMessages(session.messages);
    renderSidebarSessions();

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
    if (confirm(t('clearAllConfirm'))) {
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
        dom.systemStatus.innerHTML = `<span class="status-dot"></span><span class="status-label">${t('systemOnline')}</span>`;
      }
    } catch {
      if (dom.systemStatus) {
        dom.systemStatus.innerHTML = `<span class="status-dot" style="background-color: var(--accent-rose); box-shadow: 0 0 8px rgba(244,63,94,0.4);"></span><span class="status-label" style="color: var(--accent-rose);">${t('systemOffline')}</span>`;
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
        <span class="session-item-title">${escapeHtml(s.title || t('chatTitleDefault'))}</span>
        <button class="session-item-del-btn" title="Xóa">✕</button>
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
      dom.sessionsToday.innerHTML = `<div class="vault-empty-hint">${t('todayEmpty')}</div>`;
    }
    if (!hasRecent) {
      dom.sessionsRecent.innerHTML = `<div class="vault-empty-hint">${t('recentEmpty')}</div>`;
    }
  }

  function renderDocumentVault() {
    if (!dom.documentVaultList) return;
    dom.documentVaultList.innerHTML = '';

    if (!state.ocrVault || state.ocrVault.length === 0) {
      dom.documentVaultList.innerHTML = `<div class="vault-empty-hint">${t('vaultEmpty')}</div>`;
      return;
    }

    state.ocrVault.forEach((doc) => {
      const item = document.createElement('div');
      item.className = 'vault-item';
      item.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        </svg>
        <span class="vault-item-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <span class="vault-item-badge">${doc.pageCount || 1} ${t('drawerPages').toLowerCase()}</span>
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
    const runningTitle = t('thinkingProcessRunning');
    const doneTitle = `${t('thinkingProcessDone')} (${steps.length})`;

    header.innerHTML = `
      <div class="thinking-title">
        ${isLive ? '<div class="thinking-spinner"></div>' : '<span>🧠</span>'}
        <span class="thinking-title-text">${isLive ? runningTitle : doneTitle}</span>
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
          <button class="copy-code-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.textContent='${t('copiedCode')}'; setTimeout(()=>this.textContent='${t('copyCode')}', 2000);">${t('copyCode')}</button>
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

      rows.forEach((row) => {
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

    // Paragraphs
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

    // Citations highlighting [Article 468 Civil Code 2015] / [Điều 468 BLDS 2015]
    text = text.replace(/\[((?:Điều|Khoản|Bộ luật|Luật|Án lệ|Nghị định|Article|Clause|Civil Code|Commercial Law)[^\]]+)\]/g, '<span class="citation-chip">[$1]</span>');

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

    let session = getCurrentSession();
    if (!session) {
      createNewSession(false);
      session = getCurrentSession();
    }

    if (session.messages.length === 0) {
      const cleanTitle = (text || file.name).slice(0, 36).replace(/[\r\n]+/g, ' ');
      session.title = cleanTitle;
      if (dom.currentChatTitle) dom.currentChatTitle.textContent = cleanTitle;
    }

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

    dom.promptInput.value = '';
    dom.promptInput.style.height = 'auto';
    clearAttachment();

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

    const liveAccordion = createThinkingAccordionElement(thinkingSteps, true);
    liveAccordion.classList.add('open');
    wrapperEl.insertBefore(liveAccordion, bubbleEl);

    const formData = new FormData();
    if (text) formData.append('message', text);
    if (session.id) formData.append('session_id', session.id);
    if (file) formData.append('file', file);
    formData.append('lang', state.lang);

    state.activeController = new AbortController();

    try {
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        body: formData,
        signal: state.activeController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const evtStr of events) {
          const line = evtStr.trim();
          if (!line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'step') {
              const langMap = STEP_METADATA[state.lang] || STEP_METADATA.vi;
              const stepInfo = langMap[data.step] || { label: data.step, icon: '⚡' };
              const stepMsg = data.message || stepInfo.label;

              thinkingSteps.push({
                step: data.step,
                status: data.status || 'running',
                message: stepMsg,
              });

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
                bubbleEl.innerHTML = `<p>${data.message || 'Audit completed.'}</p>`;
              }

            } else if (data.type === 'error') {
              bubbleEl.innerHTML = `<div style="color: var(--accent-rose); font-weight: 500;">❌ ${escapeHtml(data.message)}</div>`;
            }
          } catch (e) {
            console.warn('SSE parsing issue:', e, line);
          }
        }
      }

      // Finish thinking accordion
      const titleText = liveAccordion.querySelector('.thinking-title-text');
      const spinner = liveAccordion.querySelector('.thinking-spinner');
      if (spinner) spinner.remove();
      if (titleText) {
        titleText.textContent = `${t('thinkingProcessDone')} (${thinkingSteps.length})`;
      }
      liveAccordion.classList.remove('open');

      session.messages.push(assistantMsg);
      saveSessions();
    } catch (err) {
      if (err.name === 'AbortError') {
        bubbleEl.innerHTML += `<p style="color: var(--text-muted); font-style: italic;">[${state.lang === 'vi' ? 'Đã dừng phản hồi' : 'Stopped generating'}]</p>`;
      } else {
        bubbleEl.innerHTML = `<div style="color: var(--accent-rose);">⚠️ ${escapeHtml(err.message)}</div>`;
      }
    } finally {
      setGeneratingState(false);
      state.activeController = null;
      scrollToBottom();
    }
  }

  function renderAuditReportIntoBubble(rep, bubbleEl, wrapperEl) {
    let html = '';
    const score = rep.overall_score || 85;
    const risk = rep.risk_level || (state.lang === 'vi' ? 'Trung bình' : 'Medium');
    const issuesCount = (rep.vulnerabilities || []).length;
    const scoreClass = score >= 80 ? 'green' : score >= 60 ? 'yellow' : 'red';

    html += `
      <div class="health-score-card">
        <div class="score-circle ${scoreClass}">
          <span class="score-num">${score}</span>
          <span class="score-max">/100</span>
        </div>
        <div class="score-meta">
          <div class="score-title">${t('healthScoreTitle')}: ${score}/100 (${risk})</div>
          <div class="score-desc">${state.lang === 'vi' ? `Phát hiện ${issuesCount} điểm rủi ro và các bẫy điều khoản cần xem xét sửa đổi.` : `Identified ${issuesCount} risk points and contract traps requiring revision.`}</div>
        </div>
      </div>
    `;

    if (rep.executive_summary) {
      html += `<h3>${state.lang === 'vi' ? 'Tóm tắt Đánh giá Pháp lý' : 'Executive Legal Summary'}</h3><p>${escapeHtml(rep.executive_summary)}</p>`;
    }

    if (rep.vulnerabilities && rep.vulnerabilities.length > 0) {
      html += `<h3>${t('vulnerabilitiesTitle')}</h3>`;
      rep.vulnerabilities.forEach((v, idx) => {
        const sevClass = v.severity === 'HIGH' ? 'red' : v.severity === 'MEDIUM' ? 'yellow' : 'green';
        html += `
          <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 12px 16px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <strong>#${idx + 1}. ${escapeHtml(v.clause_id || (state.lang === 'vi' ? 'Điều khoản' : 'Clause'))}</strong>
              <span class="score-circle ${sevClass}" style="width: auto; height: auto; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem;">${t('levelRisk')}: ${v.severity}</span>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 6px;">${escapeHtml(v.issue_description)}</p>
            ${v.legal_ground ? `<p style="font-size: 0.8rem; color: var(--text-code);"><strong>${state.lang === 'vi' ? 'Căn cứ' : 'Statute'}:</strong> ${escapeHtml(v.legal_ground)}</p>` : ''}
          </div>
        `;
      });
    }

    if (rep.redline_recommendations && rep.redline_recommendations.length > 0) {
      html += `<h3>${t('redlineTitle')}</h3>`;
      rep.redline_recommendations.forEach((r) => {
        html += `
          <div style="margin: 12px 0;">
            <div style="font-weight: 600; font-size: 0.88rem; color: var(--accent-cyan); margin-bottom: 4px;">${escapeHtml(r.clause_id || 'Redline')}</div>
            <div class="diff-container">
              <div class="diff-line deletion">- ${escapeHtml(r.original_text || '')}</div>
              <div class="diff-line addition">+ ${escapeHtml(r.suggested_revision || '')}</div>
            </div>
            ${r.rationale ? `<p style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">${escapeHtml(r.rationale)}</p>` : ''}
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
        dom.sendBtn.title = t('stopBtnTitle');
      } else {
        dom.sendIcon.style.display = 'block';
        dom.stopIcon.style.display = 'none';
        dom.sendBtn.title = t('sendBtnTitle');
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
        <button class="attachment-remove-btn" id="remove-attachment-btn" title="✕">✕</button>
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
          <span>${state.lang === 'vi' ? 'Đang bóc tách OCR văn bản & bảng biểu...' : 'Extracting OCR text & tables...'}</span>
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
        throw new Error(`OCR error: ${res.status}`);
      }

      const data = await res.json();
      if (dom.drawerPageCount) dom.drawerPageCount.textContent = data.page_count || 1;
      if (dom.drawerTableCount) dom.drawerTableCount.textContent = (data.tables || []).length;
      if (dom.drawerCharCount) dom.drawerCharCount.textContent = data.char_count || 0;
      if (dom.drawerMarkdownPreview) {
        dom.drawerMarkdownPreview.textContent = data.markdown || '';
      }

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
      alert(t('exportEmpty'));
      return;
    }

    let md = `${t('exportHeader')}\n`;
    md += `**${state.lang === 'vi' ? 'Chủ đề' : 'Topic'}:** ${session.title}\n`;
    md += `**${state.lang === 'vi' ? 'Thời gian' : 'Date'}:** ${new Date(session.createdAt).toLocaleString(state.lang === 'vi' ? 'vi-VN' : 'en-US')}\n\n`;
    md += `---\n\n`;

    session.messages.forEach((m) => {
      const roleName = m.role === 'user' ? t('userLabel') : t('assistantLabel');
      md += `### ${roleName}\n\n`;
      if (m.attachedFile) {
        md += `> ${t('attachedFileLabel')}: **${m.attachedFile.name}**\n\n`;
      }
      md += `${m.content}\n\n`;
      if (m.citations && m.citations.length > 0) {
        md += `**${t('legalGroundsLabel')}:** ${m.citations.join(' | ')}\n\n`;
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
    // Language Toggle
    if (dom.langToggleBtn) {
      dom.langToggleBtn.addEventListener('click', toggleLanguage);
    }

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

    // Input auto-expand & Enter to send
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

    // Prompt Chips Click Handlers
    dom.promptChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        if (chip.id === 'trigger-ocr-card') {
          if (dom.ocrUploadInput) dom.ocrUploadInput.click();
          return;
        }

        const promptKey = chip.getAttribute('data-prompt-key');
        let textToSend = '';

        if (promptKey === 'chipAudit') {
          textToSend = t('chipAuditPrompt');
        } else if (promptKey === 'chipLookup') {
          textToSend = t('chipLookupPrompt');
        } else if (promptKey === 'chipTable') {
          textToSend = t('chipTablePrompt');
        }

        if (textToSend && dom.promptInput) {
          dom.promptInput.value = textToSend;
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
        if (!text || text.includes(t('drawerEmpty'))) return;
        navigator.clipboard.writeText(text);
        dom.copyOcrTextBtn.textContent = t('drawerCopied');
        setTimeout(() => (dom.copyOcrTextBtn.textContent = t('drawerCopyBtn')), 2000);
      });
    }

    if (dom.sendOcrToChatBtn && dom.drawerMarkdownPreview && dom.promptInput) {
      dom.sendOcrToChatBtn.addEventListener('click', () => {
        const text = dom.drawerMarkdownPreview.innerText;
        if (!text || text.includes(t('drawerEmpty'))) return;
        closeDrawer();
        dom.promptInput.value = `${t('ocrPromptPrefix')}${text}`;
        dom.promptInput.focus();
        dom.promptInput.style.height = 'auto';
        dom.promptInput.style.height = Math.min(dom.promptInput.scrollHeight, 180) + 'px';
      });
    }

    // Drag and Drop Files
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
