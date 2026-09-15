(() => {
  "use strict";

  const ROLES = ["runner", "client_yaml", "edge_yaml", "server_yaml"];
  const $ = id => document.getElementById(id);
  const state = {
    lang: "zh",
    templateName: "",
    paths: [],
    baseCategories: [],
    categories: [],
    aliases: new Map(),
    baseAliases: new Map(),
    selections: Object.fromEntries(ROLES.map(role => [role, []])),
    extraHeader: "",
    files: new Map(),
    directoryLoaded: false,
    directoryFileCount: 0,
    fmRoot: null,
    fmTree: null,
    fmExpanded: new Set(),
    fmCurrent: null,
    fmDirty: false,
  };

  const text = {
    zh: {
      title: "FL YAML 组合生成器", genTab: "组合生成器", fmTab: "YAML 管理",
      loadTemplate: "载入模板 YAML", loadYamlsDir: "载入 yamls 目录",
      yamlsDirLoaded: "yamls 目录已读取", copy: "复制", download: "下载 YAML",
      pathConfig: "路径配置", preview: "预览", selected: "已选择（有序）",
      available: "可用组件（点击添加）", emptyHint: "暂无组件",
      defaultTpl: "使用内置默认模板", templateLoaded: n => `已载入：${n}`,
      openWritableDir: "打开/刷新可编辑目录", fmSave: "保存", fmRename: "重命名",
      fmDelete: "删除", fmSelectFile: "选择左侧文件进行编辑",
      fmEmpty: "点击“打开/刷新可编辑目录”选择 yamls 文件夹",
      fmUnsupported: "此浏览器不支持可写目录，已切换到只读目录选择。",
      copied: "已复制", copyFailed: "复制失败", downloaded: "已下载",
      directoryRead: n => `已读取 ${n} 个 YAML 文件`, saved: "已保存",
      renamed: "已重命名", deleted: "已删除", writeFailed: e => `操作失败：${e}`,
      unsaved: "当前文件尚未保存，继续会丢失修改。是否继续？",
      deleteConfirm: n => `确定删除 ${n}？`, renamePrompt: "新文件名：",
      noContent: "尚未读取该文件。请先载入 yamls 目录。",
      fileLabel: n => `文件：${n}`, addTip: r => `添加到 ${r}`,
      guideTitle: "先载入 YAML 目录",
      guideIntro: "工具需要读取磁盘中的 YAML 文件，才能建立完整的可用组件列表和注册表。",
      guideSteps: ["点击页面顶部的“载入 yamls 目录”。", "选择项目中的 src/yamls 文件夹；浏览器询问时允许读取。", "如需修改已有组合，再点击“载入模板 YAML”（可选）。", "从左侧可用组件中选择配置，右侧会自动生成完整 YAML。", "最后复制或下载生成结果。"],
      guideTip: "提示：目录内容发生变化后，再次载入同一目录即可刷新。",
    },
    en: {
      title: "FL YAML Combination Generator", genTab: "Generator", fmTab: "YAML Manager",
      loadTemplate: "Load template YAML", loadYamlsDir: "Load yamls directory",
      yamlsDirLoaded: "yamls directory loaded", copy: "Copy", download: "Download YAML",
      pathConfig: "Path configuration", preview: "Preview", selected: "Selected (ordered)",
      available: "Available components (click to add)", emptyHint: "No components",
      defaultTpl: "Using embedded default template", templateLoaded: n => `Loaded: ${n}`,
      openWritableDir: "Open/refresh writable directory", fmSave: "Save", fmRename: "Rename",
      fmDelete: "Delete", fmSelectFile: "Select a file on the left to edit",
      fmEmpty: "Click “Open/refresh writable directory” and select the yamls folder",
      fmUnsupported: "Writable directories are unsupported; using read-only selection.",
      copied: "Copied", copyFailed: "Copy failed", downloaded: "Downloaded",
      directoryRead: n => `Read ${n} YAML files`, saved: "Saved", renamed: "Renamed",
      deleted: "Deleted", writeFailed: e => `Operation failed: ${e}`,
      unsaved: "The current file has unsaved changes. Continue?",
      deleteConfirm: n => `Delete ${n}?`, renamePrompt: "New filename:",
      noContent: "File not loaded. Load the yamls directory first.",
      fileLabel: n => `File: ${n}`, addTip: r => `Add to ${r}`,
      guideTitle: "Load the YAML directory first",
      guideIntro: "The tool must read YAML files from disk before it can build the complete component list and registry.",
      guideSteps: ["Click “Load yamls directory” at the top of the page.", "Select the project's src/yamls folder and allow read access when prompted.", "Optionally load a template YAML to edit an existing combination.", "Choose components on the left; the complete YAML is generated on the right.", "Copy or download the generated result."],
      guideTip: "Tip: load the same directory again to refresh changes made on disk.",
    }
  };
  const t = (key, arg) => {
    const value = text[state.lang][key] ?? key;
    return typeof value === "function" ? value(arg) : value;
  };

  function normalizeSlash(value) { return String(value || "").replace(/\\/g, "/"); }
  function trimPath(value) { return normalizeSlash(value).replace(/\/+$/, ""); }
  function pathDirectory(value) {
    const parts = trimPath(value).split("/").filter(Boolean);
    return parts[parts.length - 1] || "";
  }
  function cleanLogical(value) {
    return value.replace(/\.(yaml|yml)$/i, "").replace(/[^A-Za-z0-9_-]+/g, "_");
  }
  function cloneCategories(source) {
    return source.map(category => ({ cat: category.cat, names: [...category.names] }));
  }

  function parseTemplate(source) {
    const lines = String(source || "").replace(/^\uFEFF/, "").split(/\r?\n/);
    const paths = [];
    const categories = [];
    const aliases = new Map();
    const selections = Object.fromEntries(ROLES.map(role => [role, []]));
    const extra = [];
    const pathRe = /^(yaml_folder_(.+?)_path):\s*(.*?)\s*$/;
    const filesRe = /^yaml_folder_(.+?)_files:\s*(?:\[\])?\s*$/;
    const itemRe = /^-\s*([^:]+?):\s*(.+?)\s*$/;
    const roleRe = /^\s{2}(runner|client_yaml|edge_yaml|server_yaml):\s*(?:\[\])?\s*$/;
    const roleItemRe = /^\s{2,}-\s*(.+?)\s*$/;
    let inCombination = false;
    let currentCategory = null;
    let currentRole = null;

    for (const line of lines) {
      if (/^yaml_combination:\s*$/.test(line)) {
        inCombination = true;
        currentCategory = null;
        continue;
      }
      if (inCombination) {
        const roleMatch = line.match(roleRe);
        if (roleMatch) { currentRole = roleMatch[1]; continue; }
        const roleItem = line.match(roleItemRe);
        if (roleItem && currentRole) selections[currentRole].push(roleItem[1].trim());
        continue;
      }

      const pathMatch = line.match(pathRe);
      if (pathMatch) {
        paths.push({ key: pathMatch[1], cat: pathMatch[2], value: pathMatch[3] });
        currentCategory = null;
        continue;
      }
      const filesMatch = line.match(filesRe);
      if (filesMatch) {
        currentCategory = { cat: filesMatch[1], names: [] };
        categories.push(currentCategory);
        continue;
      }
      const itemMatch = currentCategory && line.match(itemRe);
      if (itemMatch) {
        const filename = itemMatch[1].trim();
        const logical = itemMatch[2].trim();
        currentCategory.names.push(logical);
        aliases.set(logical, filename);
        continue;
      }
      currentCategory = null;
      extra.push(line);
    }

    for (const path of paths) {
      if (!categories.some(category => category.cat === path.cat)) {
        categories.push({ cat: path.cat, names: [] });
      }
    }
    return {
      paths, categories, aliases, selections,
      extraHeader: extra.join("\n").replace(/^\s+|\s+$/g, ""),
    };
  }

  function loadTemplate(source, name = "") {
    const parsed = parseTemplate(source);
    state.paths = parsed.paths;
    state.baseCategories = cloneCategories(parsed.categories);
    state.baseAliases = new Map(parsed.aliases);
    state.selections = parsed.selections;
    state.extraHeader = parsed.extraHeader;
    state.templateName = name;
    rebuildAvailableComponents();
    renderAll();
  }

  function uniqueLogical(preferred, used) {
    let result = preferred;
    let suffix = 2;
    while (used.has(result)) result = `${preferred}_${suffix++}`;
    return result;
  }

  function rebuildAvailableComponents() {
    state.categories = cloneCategories(state.baseCategories);
    state.aliases = new Map(state.baseAliases);
    if (!state.directoryLoaded) return;

    const dirToCategory = new Map();
    for (const path of state.paths) {
      const directory = pathDirectory(path.value);
      if (directory) dirToCategory.set(directory, path.cat);
    }
    const diskByCategory = new Map();
    for (const [relativePath, file] of state.files) {
      const parts = normalizeSlash(relativePath).split("/").filter(Boolean);
      if (parts.length < 2) continue;
      const category = dirToCategory.get(parts[parts.length - 2]);
      if (!category) continue;
      if (!diskByCategory.has(category)) diskByCategory.set(category, []);
      diskByCategory.get(category).push({ filename: parts[parts.length - 1], file });
    }

    for (const [cat, diskFiles] of diskByCategory) {
      let category = state.categories.find(item => item.cat === cat);
      if (!category) {
        category = { cat, names: [] };
        state.categories.push(category);
      }
      const oldAliasByFile = new Map();
      for (const name of category.names) oldAliasByFile.set(state.baseAliases.get(name), name);
      category.names = [];
      const used = new Set(state.aliases.keys());
      diskFiles.sort((a, b) => a.filename.localeCompare(b.filename));
      for (const diskFile of diskFiles) {
        let logical = oldAliasByFile.get(diskFile.filename);
        if (!logical) logical = uniqueLogical(`${cat}_${cleanLogical(diskFile.filename)}`, used);
        used.add(logical);
        category.names.push(logical);
        state.aliases.set(logical, diskFile.filename);
      }
    }
  }

  function normalizeSelectedPath(rawPath) {
    const parts = normalizeSlash(rawPath).split("/").filter(Boolean);
    const yamlsIndex = parts.lastIndexOf("yamls");
    if (yamlsIndex >= 0) return parts.slice(yamlsIndex + 1).join("/");
    const configuredDirs = new Set(state.paths.map(path => pathDirectory(path.value)));
    if (configuredDirs.has(parts[0])) return parts.join("/");
    return parts.slice(1).join("/");
  }

  async function loadFileList(fileList) {
    const yamlFiles = Array.from(fileList).filter(file => /\.ya?ml$/i.test(file.name));
    const next = new Map();
    await Promise.all(yamlFiles.map(async file => {
      const relative = normalizeSelectedPath(file.webkitRelativePath || file.name);
      if (relative) next.set(relative, { name: file.name, text: await file.text(), file });
    }));
    state.files = next;
    state.directoryLoaded = true;
    state.directoryFileCount = yamlFiles.length;
    rebuildAvailableComponents();
    renderAll();
    toast(t("directoryRead", yamlFiles.length));
  }

  function buildYaml() {
    const out = [];
    for (const path of state.paths) out.push(`${path.key}: ${path.value}`);
    if (state.paths.length) out.push("");
    if (state.extraHeader) { out.push(state.extraHeader, ""); }
    for (const category of state.categories) {
      if (!category.names.length) {
        out.push(`yaml_folder_${category.cat}_files: []`, "");
        continue;
      }
      out.push(`yaml_folder_${category.cat}_files:`);
      for (const logical of category.names) {
        out.push(`- ${state.aliases.get(logical) || logical + ".yaml"}: ${logical}`);
      }
      out.push("");
    }
    out.push("yaml_combination:");
    ROLES.forEach((role, index) => {
      const names = state.selections[role] || [];
      if (!names.length) out.push(`  ${role}: []`);
      else {
        out.push(`  ${role}:`);
        for (const name of names) out.push(`  - ${name}`);
      }
      if (index < ROLES.length - 1) out.push("");
    });
    return out.join("\n") + "\n";
  }

  function renderAll() {
    renderPaths();
    renderRoles();
    renderPreview();
    updateLabels();
  }

  function updateLabels() {
    document.title = t("title");
    document.querySelectorAll("[data-i18n]").forEach(element => {
      const key = element.dataset.i18n;
      if (typeof text[state.lang][key] === "string") element.textContent = t(key);
    });
    $("btn-lang").textContent = state.lang === "zh" ? "EN" : "中文";
    $("tpl-info").textContent = state.templateName ? t("templateLoaded", state.templateName) : t("defaultTpl");
    const label = $("yamls-dir-label");
    label.classList.toggle("loaded", state.directoryLoaded);
    label.querySelector("span").textContent = state.directoryLoaded
      ? `${t("yamlsDirLoaded")} (${state.directoryFileCount})` : t("loadYamlsDir");
    $("path-count").textContent = `(${state.paths.length})`;
  }

  function renderPaths() {
    const root = $("path-grid");
    root.innerHTML = "";
    state.paths.forEach((path, index) => {
      const row = document.createElement("div"); row.className = "path-row";
      const label = document.createElement("label"); label.textContent = path.key; label.title = path.key;
      const input = document.createElement("input"); input.value = path.value;
      input.addEventListener("change", () => {
        state.paths[index].value = input.value.trim();
        rebuildAvailableComponents(); renderAll();
      });
      row.append(label, input); root.appendChild(row);
    });
  }

  function button(label, handler) {
    const element = document.createElement("button");
    element.className = "mini ghost"; element.textContent = label; element.onclick = handler;
    return element;
  }

  function renderRoles() {
    const root = $("roles"); root.innerHTML = "";
    for (const role of ROLES) {
      const card = document.createElement("div"); card.className = "role-card";
      const heading = document.createElement("h2");
      heading.innerHTML = `<span>${role}</span><span class="cnt">(${state.selections[role].length})</span>`;
      const body = document.createElement("div"); body.className = "role-body";
      const selected = document.createElement("div"); selected.className = "selected";
      const selectedTitle = document.createElement("div"); selectedTitle.className = "col-title"; selectedTitle.textContent = t("selected");
      selected.appendChild(selectedTitle);
      if (!state.selections[role].length) {
        const empty = document.createElement("div"); empty.className = "empty-hint"; empty.textContent = t("emptyHint"); selected.appendChild(empty);
      } else {
        const list = document.createElement("ul"); list.className = "sel-list";
        state.selections[role].forEach((name, index) => {
          const item = document.createElement("li");
          const label = document.createElement("span"); label.className = "nm"; label.textContent = name;
          label.onclick = () => openPreview(name);
          const badge = document.createElement("span"); badge.className = "catbadge"; badge.textContent = categoryOf(name);
          item.append(label, badge,
            button("↑", () => moveSelection(role, index, -1)),
            button("↓", () => moveSelection(role, index, 1)),
            button("×", () => { state.selections[role].splice(index, 1); renderRoles(); renderPreview(); }));
          list.appendChild(item);
        });
        selected.appendChild(list);
      }

      const available = document.createElement("div"); available.className = "available";
      const availableTitle = document.createElement("div"); availableTitle.className = "col-title"; availableTitle.textContent = t("available");
      const groups = document.createElement("div"); groups.className = "avail-groups";
      for (const category of state.categories) {
        if (!category.names.length) continue;
        const group = document.createElement("div"); group.className = "avail-cat";
        const name = document.createElement("div"); name.className = "cat-name"; name.textContent = category.cat;
        group.appendChild(name);
        for (const logical of category.names) {
          const chip = document.createElement("span");
          chip.className = `chip chip-preview${state.selections[role].includes(logical) ? " used" : ""}`;
          chip.textContent = logical; chip.title = t("addTip", role);
          chip.onclick = event => {
            if (event.ctrlKey || event.metaKey) openPreview(logical);
            else if (!state.selections[role].includes(logical)) {
              state.selections[role].push(logical); renderRoles(); renderPreview();
            }
          };
          chip.oncontextmenu = event => { event.preventDefault(); openPreview(logical); };
          group.appendChild(chip);
        }
        groups.appendChild(group);
      }
      available.append(availableTitle, groups); body.append(selected, available); card.append(heading, body); root.appendChild(card);
    }
  }

  function categoryOf(logical) {
    return state.categories.find(category => category.names.includes(logical))?.cat || "?";
  }
  function moveSelection(role, index, delta) {
    const target = index + delta;
    if (target < 0 || target >= state.selections[role].length) return;
    [state.selections[role][index], state.selections[role][target]] = [state.selections[role][target], state.selections[role][index]];
    renderRoles(); renderPreview();
  }
  function findFile(logical) {
    const cat = categoryOf(logical);
    const filename = state.aliases.get(logical);
    const directory = pathDirectory(state.paths.find(path => path.cat === cat)?.value);
    return state.files.get(`${directory}/${filename}`) || [...state.files.values()].find(item => item.name === filename);
  }
  function openPreview(logical) {
    const file = findFile(logical);
    $("modal-title").textContent = logical;
    $("modal-cat").textContent = categoryOf(logical);
    $("modal-content").textContent = file?.text ?? t("noContent");
    $("modal-foot").textContent = t("fileLabel", state.aliases.get(logical) || "");
    $("modal-overlay").classList.add("open");
  }
  function renderPreview() {
    const guide = $("preview-guide");
    const preview = $("preview");
    if (!state.directoryLoaded) {
      const steps = t("guideSteps").map(step => `<li>${escapeHtml(step)}</li>`).join("");
      guide.innerHTML = `<div class="preview-guide-card"><h3>${escapeHtml(t("guideTitle"))}</h3>` +
        `<p>${escapeHtml(t("guideIntro"))}</p><ol>${steps}</ol>` +
        `<p style="margin-top:16px"><code>${escapeHtml(t("guideTip"))}</code></p></div>`;
      guide.style.display = "flex";
      preview.style.display = "none";
    } else {
      guide.style.display = "none";
      preview.style.display = "block";
      preview.textContent = buildYaml();
    }
    let counts = document.querySelector(".pv-counts");
    if (!counts) { counts = document.createElement("span"); counts.className = "pv-counts"; document.querySelector(".pv-head").appendChild(counts); }
    counts.textContent = ROLES.map(role => `${role}:${state.selections[role].length}`).join("  ");
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[character]);
  }
  function toast(message) {
    const element = $("toast"); element.textContent = message; element.classList.add("show");
    clearTimeout(toast.timer); toast.timer = setTimeout(() => element.classList.remove("show"), 1800);
  }

  async function readTree(handle, prefix = "") {
    const path = prefix ? `${prefix}/${handle.name}` : handle.name;
    const node = { name: handle.name, path, kind: handle.kind, handle, children: [] };
    if (handle.kind === "directory") {
      const entries = [];
      for await (const entry of handle.values()) entries.push(entry);
      entries.sort((a, b) => a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === "directory" ? -1 : 1);
      for (const entry of entries) {
        if (entry.kind === "directory" || /\.ya?ml$/i.test(entry.name)) node.children.push(await readTree(entry, path));
      }
    }
    return node;
  }
  function flattenFiles(node, output = []) {
    if (node.kind === "file") output.push(node);
    else for (const child of node.children) flattenFiles(child, output);
    return output;
  }
  async function refreshWritableDirectory(chooseNew = false) {
    if (!window.showDirectoryPicker) { toast(t("fmUnsupported")); $("yamls-dir-input").click(); return; }
    try {
      if (!state.fmRoot || chooseNew) state.fmRoot = await window.showDirectoryPicker({ mode: "readwrite" });
      state.fmTree = await readTree(state.fmRoot);
      state.fmExpanded.add(state.fmTree.path);
      const fileNodes = flattenFiles(state.fmTree);
      const next = new Map();
      await Promise.all(fileNodes.map(async node => {
        const file = await node.handle.getFile();
        const relative = normalizeSelectedPath(node.path);
        next.set(relative, { name: file.name, text: await file.text(), file, node });
      }));
      state.files = next; state.directoryLoaded = true; state.directoryFileCount = fileNodes.length;
      rebuildAvailableComponents(); renderAll(); renderTree();
      toast(t("directoryRead", fileNodes.length));
    } catch (error) { if (error.name !== "AbortError") toast(t("writeFailed", error.message)); }
  }
  function renderTree() {
    const root = $("fm-tree"); root.innerHTML = "";
    if (!state.fmTree) { const empty = document.createElement("div"); empty.className = "fm-empty"; empty.textContent = t("fmEmpty"); root.appendChild(empty); return; }
    root.appendChild(renderTreeNode(state.fmTree));
  }
  function renderTreeNode(node) {
    const wrapper = document.createElement("div"); wrapper.className = "fm-node";
    const row = document.createElement("div");
    if (node.kind === "directory") {
      const open = state.fmExpanded.has(node.path); row.className = `fm-dir-row${open ? " open" : ""}`;
      row.innerHTML = `<span class="fm-arrow">▶</span><span class="fm-dname"></span><span class="fm-add">＋</span>`;
      row.querySelector(".fm-dname").textContent = node.name;
      row.querySelector(".fm-add").onclick = event => { event.stopPropagation(); createYaml(node); };
      row.onclick = () => { open ? state.fmExpanded.delete(node.path) : state.fmExpanded.add(node.path); renderTree(); };
      wrapper.appendChild(row);
      const children = document.createElement("div"); children.className = `fm-children${open ? "" : " collapsed"}`;
      node.children.forEach(child => children.appendChild(renderTreeNode(child))); wrapper.appendChild(children);
    } else {
      row.className = `fm-file-row${state.fmCurrent === node ? " active" : ""}`;
      row.textContent = node.name; row.onclick = () => openEditor(node); wrapper.appendChild(row);
    }
    return wrapper;
  }
  async function openEditor(node) {
    if (state.fmDirty && !confirm(t("unsaved"))) return;
    const file = await node.handle.getFile();
    state.fmCurrent = node; state.fmDirty = false;
    $("fm-textarea").value = await file.text(); $("fm-textarea").style.display = "";
    $("fm-placeholder").style.display = "none"; $("fm-cur").textContent = node.path;
    updateEditorButtons(); renderTree();
  }
  function updateEditorButtons() {
    $("fm-save").disabled = !state.fmCurrent || !state.fmDirty;
    $("fm-rename").disabled = !state.fmCurrent; $("fm-delete").disabled = !state.fmCurrent;
  }
  async function saveEditor() {
    if (!state.fmCurrent) return;
    try {
      const writable = await state.fmCurrent.handle.createWritable();
      await writable.write($("fm-textarea").value); await writable.close();
      state.fmDirty = false; updateEditorButtons(); await refreshWritableDirectory(); toast(t("saved"));
    } catch (error) { toast(t("writeFailed", error.message)); }
  }
  async function renameEditor() {
    const node = state.fmCurrent; if (!node) return;
    let name = prompt(t("renamePrompt"), node.name); if (!name) return;
    if (!/\.ya?ml$/i.test(name)) name += ".yaml";
    try {
      const parentPath = node.path.split("/").slice(1, -1);
      let parent = state.fmRoot; for (const part of parentPath) parent = await parent.getDirectoryHandle(part);
      const next = await parent.getFileHandle(name, { create: true });
      const writable = await next.createWritable(); await writable.write($("fm-textarea").value); await writable.close();
      await parent.removeEntry(node.name); state.fmCurrent = null; state.fmDirty = false;
      await refreshWritableDirectory(); toast(t("renamed"));
    } catch (error) { toast(t("writeFailed", error.message)); }
  }
  async function deleteEditor() {
    const node = state.fmCurrent; if (!node || !confirm(t("deleteConfirm", node.name))) return;
    try {
      const parentPath = node.path.split("/").slice(1, -1);
      let parent = state.fmRoot; for (const part of parentPath) parent = await parent.getDirectoryHandle(part);
      await parent.removeEntry(node.name); state.fmCurrent = null; state.fmDirty = false;
      $("fm-textarea").style.display = "none"; $("fm-placeholder").style.display = "flex";
      await refreshWritableDirectory(); toast(t("deleted"));
    } catch (error) { toast(t("writeFailed", error.message)); }
  }
  async function createYaml(directoryNode) {
    let name = prompt(t("renamePrompt"), "new_config.yaml");
    if (!name) return;
    if (!/\.ya?ml$/i.test(name)) name += ".yaml";
    try {
      const handle = await directoryNode.handle.getFileHandle(name, { create: true });
      const writable = await handle.createWritable(); await writable.write(""); await writable.close();
      state.fmExpanded.add(directoryNode.path); await refreshWritableDirectory(); toast(t("saved"));
    } catch (error) { toast(t("writeFailed", error.message)); }
  }

  function wireUi() {
    $("btn-lang").onclick = () => { state.lang = state.lang === "zh" ? "en" : "zh"; renderAll(); renderTree(); };
    $("tpl-file").onchange = async event => { const file = event.target.files[0]; if (file) loadTemplate(await file.text(), file.name); event.target.value = ""; };
    $("yamls-dir-input").onchange = async event => { await loadFileList(event.target.files); event.target.value = ""; };
    $("btn-copy").onclick = async () => { try { await navigator.clipboard.writeText(buildYaml()); toast(t("copied")); } catch { toast(t("copyFailed")); } };
    $("btn-download").onclick = () => { const url = URL.createObjectURL(new Blob([buildYaml()], { type: "text/yaml" })); const a = document.createElement("a"); a.href = url; a.download = "fl_combination.yaml"; a.click(); URL.revokeObjectURL(url); toast(t("downloaded")); };
    $("path-toggle").onclick = () => { $("path-toggle").classList.toggle("open"); $("path-body").classList.toggle("open"); };
    $("modal-close").onclick = () => $("modal-overlay").classList.remove("open");
    $("modal-overlay").onclick = event => { if (event.target === $("modal-overlay")) $("modal-overlay").classList.remove("open"); };
    $("tab-gen").onclick = () => { $("tab-gen").classList.add("active"); $("tab-fm").classList.remove("active"); $("gen-view").style.display = "flex"; $("fm-view").style.display = "none"; };
    $("tab-fm").onclick = () => { $("tab-fm").classList.add("active"); $("tab-gen").classList.remove("active"); $("gen-view").style.display = "none"; $("fm-view").style.display = "flex"; renderTree(); };
    $("fm-open-dir").onclick = () => refreshWritableDirectory(true);
    $("fm-textarea").oninput = () => { state.fmDirty = true; updateEditorButtons(); };
    $("fm-save").onclick = saveEditor; $("fm-rename").onclick = renameEditor; $("fm-delete").onclick = deleteEditor;
    document.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s" && $("fm-view").style.display !== "none") {
        event.preventDefault(); if (state.fmCurrent && state.fmDirty) saveEditor();
      }
    });
  }

  wireUi();
  loadTemplate($("default-template").textContent);
  renderTree();
})();
