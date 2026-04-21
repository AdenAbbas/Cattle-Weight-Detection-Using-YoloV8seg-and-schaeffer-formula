/* ══════════════════════════════════════════════════════════════════
   COWVILLE — frontend interactivity (v2)
   Upload-only · no metadata · guide box alignment
   ══════════════════════════════════════════════════════════════════ */

(() => {
    // ── DOM handles ───────────────────────────────────────────────
    const dropzone    = document.getElementById('dropzone');
    const fileInput   = document.getElementById('fileInput');
    const preview     = document.getElementById('preview');
    const previewImg  = document.getElementById('previewImg');
    const clearBtn    = document.getElementById('clearBtn');

    const weighBtn    = document.getElementById('weighBtn');
    const resultEl    = document.getElementById('result');
    const weightValue = document.getElementById('weightValue');
    const verdictText = document.getElementById('verdictText');
    const needle      = document.getElementById('needle');
    const againBtn    = document.getElementById('againBtn');

    const loadingEl   = document.getElementById('loading');
    const errorEl     = document.getElementById('error');
    const errorText   = document.getElementById('errorText');

    const aboutLink   = document.getElementById('aboutLink');
    const aboutModal  = document.getElementById('aboutModal');
    const closeModal  = document.getElementById('closeModal');

    // ── State ─────────────────────────────────────────────────────
    let selectedFile = null;

    // ── File upload (click or drag-drop) ─────────────────────────
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragging');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragging');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragging');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError("That file isn't a cow photo, partner!");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            preview.classList.remove('hidden');
            dropzone.classList.add('hidden');
            weighBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        preview.classList.add('hidden');
        dropzone.classList.remove('hidden');
        weighBtn.disabled = true;
        resultEl.classList.add('hidden');
        hideError();
    });

    // ── Weigh button → API call ───────────────────────────────────
    weighBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        hideError();
        resultEl.classList.add('hidden');
        loadingEl.classList.remove('hidden');
        weighBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const res  = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (!res.ok) throw new Error(data.error || 'The farm scales broke!');

            showResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            loadingEl.classList.add('hidden');
            weighBtn.disabled = false;
        }
    });

    // ── Show result with animated scale needle ────────────────────
    function showResult(data) {
        resultEl.classList.remove('hidden');
        verdictText.textContent = data.verdict;

        const target = data.weight_kg;
        const dur    = 1200;
        const start  = performance.now();
        (function tick(now) {
            const t     = Math.min(1, (now - start) / dur);
            const eased = 1 - Math.pow(1 - t, 3);
            weightValue.textContent = Math.round(target * eased);
            if (t < 1) requestAnimationFrame(tick);
            else weightValue.textContent = target.toFixed(1);
        })(performance.now());

        // Needle: 0–600 kg → -90° to +90°
        const angle = -90 + (Math.max(0, Math.min(600, target)) / 600) * 180;
        needle.style.transition = 'none';
        needle.style.transform  = 'translateX(-50%) rotate(-90deg)';
        void needle.offsetWidth;
        needle.style.transition = 'transform 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)';
        needle.style.transform  = `translateX(-50%) rotate(${angle}deg)`;

        resultEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ── "Weigh another" → reset ───────────────────────────────────
    againBtn.addEventListener('click', () => {
        clearBtn.click();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // ── Error helpers ─────────────────────────────────────────────
    function showError(msg) {
        errorText.textContent = '⚠️ ' + msg;
        errorEl.classList.remove('hidden');
        setTimeout(hideError, 6000);
    }
    function hideError() {
        errorEl.classList.add('hidden');
    }

    // ── About modal ───────────────────────────────────────────────
    aboutLink.addEventListener('click', (e) => {
        e.preventDefault();
        aboutModal.classList.remove('hidden');
    });
    closeModal.addEventListener('click', () => aboutModal.classList.add('hidden'));
    aboutModal.addEventListener('click', (e) => {
        if (e.target === aboutModal) aboutModal.classList.add('hidden');
    });
})();
