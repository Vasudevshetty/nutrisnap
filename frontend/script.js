// API Configuration
const API_BASE_URL = "http://localhost:8000";

// Application State
const app = {
  currentTab: "name",
  loading: false,
  searchResults: [],
  autocompleteTimeout: null,
};

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

function initializeApp() {
  console.log("Initializing app..."); // Debug log

  // Check if essential elements exist
  const essentialElements = [
    "food-name-input",
    "analyze-name-btn",
    "results",
    "food-info",
    "nutrition-facts",
    "health-tags",
    "ai-insights",
  ];

  essentialElements.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      console.log(`✓ Found element: ${id}`);
    } else {
      console.error(`✗ Missing element: ${id}`);
    }
  });

  setupEventListeners();
  setupNavigation();
  checkAPIHealth();
}

function setupEventListeners() {
  console.log("Setting up event listeners..."); // Debug log

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => switchTab(e.target.dataset.tab));
  });

  // Navigation
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const section = e.target.getAttribute("href").substring(1);
      showSection(section);
    });
  });

  // Food name analysis
  const analyzeBtn = document.getElementById("analyze-name-btn");
  const foodInput = document.getElementById("food-name-input");

  console.log("Analyze button:", analyzeBtn); // Debug log
  console.log("Food input:", foodInput); // Debug log

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeByName);
  } else {
    console.error("analyze-name-btn not found!");
  }

  if (foodInput) {
    foodInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") analyzeByName();
    });
    foodInput.addEventListener("input", handleFoodNameInput);
  } else {
    console.error("food-name-input not found!");
  }

  // Image upload
  const uploadArea = document.getElementById("upload-area");
  const imageInput = document.getElementById("image-input");

  if (uploadArea && imageInput) {
    uploadArea.addEventListener("click", () => imageInput.click());
    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadArea.style.background = "rgba(46, 139, 87, 0.2)";
    });
    uploadArea.addEventListener("dragleave", (e) => {
      e.preventDefault();
      uploadArea.style.background = "rgba(46, 139, 87, 0.05)";
    });
    uploadArea.addEventListener("drop", handleImageDrop);
    imageInput.addEventListener("change", handleImageSelect);
  }

  // Manual analysis
  const manualBtn = document.getElementById("analyze-manual-btn");
  if (manualBtn) {
    manualBtn.addEventListener("click", analyzeManualData);
  }

  // Search
  const searchBtn = document.getElementById("search-btn");
  const searchInput = document.getElementById("search-input");

  if (searchBtn) {
    searchBtn.addEventListener("click", searchFoods);
  }

  if (searchInput) {
    searchInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") searchFoods();
    });
  }

  // Toast close
  const toastClose = document.querySelector(".toast-close");
  if (toastClose) {
    toastClose.addEventListener("click", hideToast);
  }
}

function setupNavigation() {
  // Show analyze section by default
  showSection("analyze");
}

function switchTab(tabName) {
  app.currentTab = tabName;

  // Update tab buttons
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  // Update tab content
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.toggle("active", content.id === `${tabName}-tab`);
  });
}

function showSection(sectionName) {
  console.log("Showing section:", sectionName); // Debug log

  // Update navigation
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle(
      "active",
      link.getAttribute("href") === `#${sectionName}`
    );
  });

  // Update sections
  document.querySelectorAll(".section").forEach((section) => {
    const shouldShow = section.id === sectionName;
    section.style.display = shouldShow ? "block" : "none";
    console.log(`Section ${section.id}: display = ${section.style.display}`); // Debug log
  });
}

function showLoading(show = true) {
  app.loading = show;
  document.getElementById("loading").style.display = show ? "flex" : "none";

  // Disable buttons
  document.querySelectorAll("button").forEach((btn) => {
    btn.disabled = show;
  });
}

function showToast(message, type = "error") {
  const toast = document.getElementById("error-toast");
  const messageEl = document.getElementById("error-message");

  messageEl.textContent = message;
  toast.className = `toast ${type}-toast`;
  toast.style.display = "flex";

  // Auto hide after 5 seconds
  setTimeout(hideToast, 5000);
}

function hideToast() {
  document.getElementById("error-toast").style.display = "none";
}

// API Functions
async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      showToast(
        "API server is not responding. Please ensure the backend is running."
      );
    }
  } catch (error) {
    showToast(
      "Cannot connect to API server. Please ensure the backend is running."
    );
  }
}

async function analyzeByName() {
  const foodName = document.getElementById("food-name-input").value.trim();

  if (!foodName) {
    showToast("Please enter a food name");
    return;
  }

  console.log("Analyzing food:", foodName); // Debug log

  showLoading(true);
  // Show results section with loading state
  document.getElementById("results").style.display = "block";
  showSection("analyze");

  // Show loading state in results
  document.getElementById("food-info").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Analyzing food...</p></div>';
  document.getElementById("nutrition-facts").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Loading nutrition facts...</p></div>';
  document.getElementById("health-tags").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Generating health tags...</p></div>';
  document.getElementById("ai-insights").innerHTML = `
    <div class="ai-insights-loading">
      <div class="loading-spinner"></div>
      <p><i class="fas fa-brain"></i> Our AI is analyzing your food and generating personalized insights...</p>
      <small>This includes health assessment, dietary recommendations, nutritional breakdown, and practical tips!</small>
    </div>
  `;

  try {
    const response = await fetch(
      `${API_BASE_URL}/analyze/food/${encodeURIComponent(foodName)}`
    );
    const data = await response.json();

    console.log("API Response:", data); // Debug log

    if (response.ok) {
      console.log("Displaying results for:", data.analysis); // Debug log

      // Ensure results section is visible
      const resultsSection = document.getElementById("results");
      if (resultsSection) {
        resultsSection.style.display = "block";
        console.log("Results section display set to block");
      } else {
        console.error("Results section not found!");
      }

      displayResults(data.analysis);

      // Scroll to results
      setTimeout(() => {
        resultsSection?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } else {
      console.error("API Error:", data); // Debug log
      showToast(data.detail || "Food not found in database");
      document.getElementById("results").style.display = "none";
    }
  } catch (error) {
    console.error("Fetch Error:", error); // Debug log
    showToast("Error analyzing food: " + error.message);
    document.getElementById("results").style.display = "none";
  } finally {
    showLoading(false);
  }
}

async function analyzeManualData() {
  const formData = {
    food_name:
      document.getElementById("manual-food-name").value || "Custom Food",
    "Caloric Value":
      parseFloat(document.getElementById("manual-calories").value) || 0,
    Protein: parseFloat(document.getElementById("manual-protein").value) || 0,
    Fat: parseFloat(document.getElementById("manual-fat").value) || 0,
    Carbohydrates:
      parseFloat(document.getElementById("manual-carbs").value) || 0,
    "Dietary Fiber":
      parseFloat(document.getElementById("manual-fiber").value) || 0,
    Sugars: parseFloat(document.getElementById("manual-sugar").value) || 0,
    Sodium: parseFloat(document.getElementById("manual-sodium").value) || 0,
    "Saturated Fats": 0,
    Calcium: 0,
    Iron: 0,
    "Vitamin C": 0,
  };

  showLoading(true);
  // Show results section with loading state
  document.getElementById("results").style.display = "block";
  showSection("analyze");

  // Show loading state in results
  document.getElementById("food-info").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Analyzing nutrition data...</p></div>';
  document.getElementById("nutrition-facts").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Processing nutrition facts...</p></div>';
  document.getElementById("health-tags").innerHTML =
    '<div class="loading-placeholder"><div class="loading-spinner"></div><p>Generating health assessment...</p></div>';
  document.getElementById("ai-insights").innerHTML = `
    <div class="ai-insights-loading">
      <div class="loading-spinner"></div>
      <p><i class="fas fa-brain"></i> Our AI is analyzing your nutrition data and generating personalized insights...</p>
      <small>This includes health assessment, dietary recommendations, nutritional breakdown, and practical tips!</small>
    </div>
  `;

  try {
    const response = await fetch(`${API_BASE_URL}/analyze/nutrition`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const data = await response.json();

    if (response.ok) {
      displayManualResults(data);
    } else {
      showToast(data.detail || "Error analyzing nutrition data");
      document.getElementById("results").style.display = "none";
    }
  } catch (error) {
    showToast("Error analyzing nutrition data: " + error.message);
    document.getElementById("results").style.display = "none";
  } finally {
    showLoading(false);
  }
}

async function searchFoods() {
  const query = document.getElementById("search-input").value.trim();

  if (!query) {
    showToast("Please enter a search term");
    return;
  }

  showLoading(true);

  try {
    const response = await fetch(
      `${API_BASE_URL}/search/food/${encodeURIComponent(query)}`
    );
    const data = await response.json();

    if (response.ok) {
      displaySearchResults(data.results);
    } else {
      showToast(data.message || "Error searching foods");
    }
  } catch (error) {
    showToast("Error searching foods: " + error.message);
  } finally {
    showLoading(false);
  }
}

// Display Functions
function displayResults(analysis) {
  console.log("displayResults called with:", analysis); // Debug log

  if (!analysis) {
    console.error("No analysis data provided");
    return;
  }

  // Food Info
  const foodInfoHtml = `
        <div class="food-info-item">
            <span class="food-info-label">Food Name</span>
            <span class="food-info-value">${analysis.name || "Unknown"}</span>
        </div>
        <div class="food-info-item">
            <span class="food-info-label">Food Group</span>
            <span class="food-info-value">${
              analysis.food_group || "Unknown"
            }</span>
        </div>
        ${
          analysis.ml_predictions?.health_category
            ? `
        <div class="food-info-item">
            <span class="food-info-label">Health Category</span>
            <span class="food-info-value">${analysis.ml_predictions.health_category}</span>
        </div>
        `
            : ""
        }
        ${
          analysis.ml_predictions?.nutrition_density
            ? `
        <div class="food-info-item">
            <span class="food-info-label">Nutrition Density Score</span>
            <span class="food-info-value">${analysis.ml_predictions.nutrition_density.toFixed(
              1
            )}</span>
        </div>
        `
            : ""
        }
    `;

  console.log("Setting food-info HTML:", foodInfoHtml); // Debug log
  const foodInfoElement = document.getElementById("food-info");
  if (foodInfoElement) {
    foodInfoElement.innerHTML = foodInfoHtml;
  } else {
    console.error("food-info element not found");
  }

  // Nutrition Facts
  const nutrition = analysis.nutrition;
  if (nutrition) {
    const nutritionHtml = `
        <div class="nutrition-grid">
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.calories || 0}</span>
                <div class="nutrition-label">Calories</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.protein || 0}g</span>
                <div class="nutrition-label">Protein</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.fat || 0}g</span>
                <div class="nutrition-label">Fat</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${
                  nutrition.carbohydrates || 0
                }g</span>
                <div class="nutrition-label">Carbs</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.fiber || 0}g</span>
                <div class="nutrition-label">Fiber</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.sugar || 0}g</span>
                <div class="nutrition-label">Sugar</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.sodium || 0}mg</span>
                <div class="nutrition-label">Sodium</div>
            </div>
        </div>
    `;

    console.log("Setting nutrition-facts HTML:", nutritionHtml); // Debug log
    const nutritionElement = document.getElementById("nutrition-facts");
    if (nutritionElement) {
      nutritionElement.innerHTML = nutritionHtml;
    } else {
      console.error("nutrition-facts element not found");
    }
  } else {
    console.error("No nutrition data found");
  }

  // Health Tags
  if (analysis.health_tags && Array.isArray(analysis.health_tags)) {
    const tagsHtml = `
        <div class="health-tags-container">
            ${analysis.health_tags
              .map(
                (tag) => `
                <span class="health-tag ${getTagClass(tag)}">${tag}</span>
            `
              )
              .join("")}
        </div>
    `;

    console.log("Setting health-tags HTML:", tagsHtml); // Debug log
    const tagsElement = document.getElementById("health-tags");
    if (tagsElement) {
      tagsElement.innerHTML = tagsHtml;
    } else {
      console.error("health-tags element not found");
    }
  } else {
    console.error("No health tags found or invalid format");
  }

  // AI Insights
  let insightsHtml = `
    <div class="ai-insights-loading">
      <div class="loading-spinner"></div>
      <p><i class="fas fa-brain"></i> Our AI is analyzing your food and generating personalized insights...</p>
      <small>This includes health assessment, dietary recommendations, nutritional breakdown, and practical tips!</small>
    </div>
  `;

  if (analysis.ai_insights) {
    console.log("AI Insights found:", analysis.ai_insights); // Debug log
    insightsHtml = `
      <div class="ai-insights-intro">
        <p><i class="fas fa-robot"></i> <strong>AI-Powered Nutrition Analysis</strong></p>
        <small>Our advanced AI has analyzed your food and generated personalized insights to help you make better dietary choices.</small>
      </div>
      <div class="ai-insights-container">
        ${Object.entries(analysis.ai_insights)
          .map(
            ([key, value]) => `
              <div class="ai-insight-card">
                <div class="ai-insight-header">
                  <i class="${getInsightIcon(key)}"></i>
                  <h4>${formatInsightTitle(key)}</h4>
                </div>
                <div class="ai-insight-content">
                  ${formatInsightContent(value)}
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  } else {
    console.log("No AI insights found"); // Debug log
  }

  console.log("Setting ai-insights HTML:", insightsHtml); // Debug log
  const insightsElement = document.getElementById("ai-insights");
  if (insightsElement) {
    insightsElement.innerHTML = insightsHtml;
  } else {
    console.error("ai-insights element not found");
  }
}

function displayManualResults(data) {
  // Similar to displayResults but adapted for manual data structure
  const foodInfoHtml = `
        <div class="food-info-item">
            <span class="food-info-label">Food Name</span>
            <span class="food-info-value">${data.food_name}</span>
        </div>
        <div class="food-info-item">
            <span class="food-info-label">Food Group</span>
            <span class="food-info-value">${data.food_group}</span>
        </div>
    `;
  document.getElementById("food-info").innerHTML = foodInfoHtml;

  // Nutrition Facts
  const nutrition = data.nutrition_data;
  const nutritionHtml = `
        <div class="nutrition-grid">
            <div class="nutrition-item">
                <span class="nutrition-value">${
                  nutrition["Caloric Value"] || 0
                }</span>
                <div class="nutrition-label">Calories</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.Protein || 0}g</span>
                <div class="nutrition-label">Protein</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.Fat || 0}g</span>
                <div class="nutrition-label">Fat</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${
                  nutrition.Carbohydrates || 0
                }g</span>
                <div class="nutrition-label">Carbs</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${
                  nutrition["Dietary Fiber"] || 0
                }g</span>
                <div class="nutrition-label">Fiber</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.Sugars || 0}g</span>
                <div class="nutrition-label">Sugar</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.Sodium || 0}mg</span>
                <div class="nutrition-label">Sodium</div>
            </div>
        </div>
    `;
  document.getElementById("nutrition-facts").innerHTML = nutritionHtml;

  // Health Tags
  const tagsHtml = `
        <div class="health-tags-container">
            ${data.health_tags
              .map(
                (tag) => `
                <span class="health-tag ${getTagClass(tag)}">${tag}</span>
            `
              )
              .join("")}
        </div>
    `;
  document.getElementById("health-tags").innerHTML = tagsHtml;

  // AI Insights
  let insightsHtml = `
    <div class="ai-insights-loading">
      <div class="loading-spinner"></div>
      <p><i class="fas fa-brain"></i> Our AI is analyzing your food and generating personalized insights...</p>
      <small>This includes health assessment, dietary recommendations, nutritional breakdown, and practical tips!</small>
    </div>
  `;

  if (
    data.ai_insights &&
    data.ai_insights.message !== "AI insights not available"
  ) {
    insightsHtml = `
      <div class="ai-insights-intro">
        <p><i class="fas fa-robot"></i> <strong>AI-Powered Nutrition Analysis</strong></p>
        <small>Our advanced AI has analyzed your food and generated personalized insights to help you make better dietary choices.</small>
      </div>
      <div class="ai-insights-container">
        ${Object.entries(data.ai_insights)
          .map(
            ([key, value]) => `
              <div class="ai-insight-card">
                <div class="ai-insight-header">
                  <i class="${getInsightIcon(key)}"></i>
                  <h4>${formatInsightTitle(key)}</h4>
                </div>
                <div class="ai-insight-content">
                  ${formatInsightContent(value)}
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }
  document.getElementById("ai-insights").innerHTML = insightsHtml;
}

function displaySearchResults(results) {
  const container = document.getElementById("search-results");

  if (!results || results.length === 0) {
    container.innerHTML = "<p>No foods found matching your search.</p>";
    return;
  }

  const resultsHtml = results
    .map(
      (food) => `
        <div class="search-result-item" onclick="analyzeSearchResult('${
          food.name
        }')">
            <div class="search-result-name">${food.name}</div>
            <div class="search-result-group">Food Group: ${
              food.food_group
            }</div>
            <div class="search-result-nutrition">
                <span>Calories: ${food.nutrition.calories}</span>
                <span>Protein: ${food.nutrition.protein}g</span>
                <span>Fat: ${food.nutrition.fat}g</span>
                <span>Carbs: ${food.nutrition.carbohydrates}g</span>
            </div>
            <div class="health-tags-container" style="margin-top: 0.5rem;">
                ${food.health_tags
                  .slice(0, 3)
                  .map(
                    (tag) => `
                    <span class="health-tag ${getTagClass(
                      tag
                    )}" style="font-size: 0.8rem;">${tag}</span>
                `
                  )
                  .join("")}
            </div>
        </div>
    `
    )
    .join("");

  container.innerHTML = resultsHtml;
}

// Helper Functions
function getTagClass(tag) {
  const healthyTags = [
    "Very Healthy",
    "Healthy",
    "High Protein",
    "High Fiber",
    "Low Fat",
    "Low Sugar",
    "Low Sodium",
    "Whole Food",
  ];
  const unhealthyTags = [
    "Less Healthy",
    "Junk Food",
    "High Sugar",
    "High Sodium",
    "High Saturated Fat",
  ];
  const moderateTags = ["Moderate", "High Fat", "High Calorie", "Low Protein"];

  if (healthyTags.some((t) => tag.includes(t))) return "healthy";
  if (unhealthyTags.some((t) => tag.includes(t))) return "unhealthy";
  if (moderateTags.some((t) => tag.includes(t))) return "moderate";
  return "neutral";
}

function formatInsightTitle(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

function getInsightIcon(key) {
  const iconMap = {
    health_assessment: "fas fa-heartbeat",
    recommendations: "fas fa-utensils",
    nutritional_breakdown: "fas fa-chart-pie",
    standards_comparison: "fas fa-balance-scale",
    dietary_tips: "fas fa-lightbulb",
  };
  return iconMap[key] || "fas fa-info-circle";
}

function formatInsightContent(content) {
  // Convert markdown-like formatting to HTML
  let formatted = content
    // Convert **text** to <strong>text</strong>
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Convert bullet points (- or •) to proper list items
    .replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>")
    // Convert line breaks to <br> tags
    .replace(/\n/g, "<br>");

  // Wrap consecutive <li> elements in <ul> tags
  formatted = formatted.replace(
    /(<li>.*?<\/li>)(?:\s*<br>\s*<li>.*?<\/li>)*/g,
    function (match) {
      return "<ul>" + match.replace(/<br>/g, "") + "</ul>";
    }
  );

  // Clean up extra <br> tags
  formatted = formatted.replace(/<br>\s*<br>/g, "<br>");

  return formatted;
}

async function analyzeSearchResult(foodName) {
  document.getElementById("food-name-input").value = foodName;
  showSection("analyze");
  await analyzeByName();
}

// Image handling functions
function handleImageDrop(e) {
  e.preventDefault();
  const uploadArea = document.getElementById("upload-area");
  uploadArea.style.background = "rgba(46, 139, 87, 0.05)";

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleImageFile(files[0]);
  }
}

function handleImageSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleImageFile(file);
  }
}

function handleImageFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Please select an image file");
    return;
  }

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById("image-preview");
    preview.innerHTML = `
            <img src="${e.target.result}" style="max-width: 100%; max-height: 300px; border-radius: 10px; margin-top: 1rem;" />
            <button onclick="analyzeImage()" class="btn-primary" style="margin-top: 1rem;">
                <i class="fas fa-search"></i>
                Analyze Image
            </button>
        `;
  };
  reader.readAsDataURL(file);

  // Store file for analysis
  app.selectedImage = file;
}

async function analyzeImage() {
  if (!app.selectedImage) {
    showToast("Please select an image first");
    return;
  }

  showLoading(true);

  try {
    const formData = new FormData();
    formData.append("file", app.selectedImage);

    const response = await fetch(`${API_BASE_URL}/predict/nutrition`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      displayResults(data.analysis || data);
      showSection("analyze");
      document.getElementById("results").style.display = "block";
    } else {
      showToast(data.detail || "Error analyzing image");
    }
  } catch (error) {
    showToast("Error analyzing image: " + error.message);
  } finally {
    showLoading(false);
  }
}

function handleFoodNameInput(e) {
  const value = e.target.value.trim();

  // Clear existing autocomplete
  clearAutocomplete();

  if (value.length >= 2) {
    // Debounce the autocomplete to avoid too many API calls
    clearTimeout(app.autocompleteTimeout);
    app.autocompleteTimeout = setTimeout(() => {
      showAutocomplete(value);
    }, 300);
  }
}

async function showAutocomplete(query) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/search/food/${encodeURIComponent(query)}`
    );
    const data = await response.json();

    if (response.ok && data.results && data.results.length > 0) {
      createAutocompleteDropdown(data.results.slice(0, 5)); // Show top 5 results
    }
  } catch (error) {
    console.error("Autocomplete error:", error);
  }
}

function createAutocompleteDropdown(foods) {
  const input = document.getElementById("food-name-input");
  const inputContainer = input.parentElement;

  // Remove existing dropdown
  clearAutocomplete();

  const dropdown = document.createElement("div");
  dropdown.className = "autocomplete-dropdown";
  dropdown.id = "autocomplete-dropdown";

  dropdown.innerHTML = foods
    .map(
      (food) => `
        <div class="autocomplete-item" onclick="selectAutocompleteItem('${food.name}')">
          <div class="autocomplete-name">${food.name}</div>
          <div class="autocomplete-group">${food.food_group}</div>
          <div class="autocomplete-nutrition">
            <span>${food.nutrition.calories} cal</span>
            <span>${food.nutrition.protein}g protein</span>
          </div>
        </div>
      `
    )
    .join("");

  inputContainer.style.position = "relative";
  inputContainer.appendChild(dropdown);

  // Add click outside to close
  setTimeout(() => {
    document.addEventListener("click", handleClickOutside);
  }, 100);
}

function selectAutocompleteItem(foodName) {
  document.getElementById("food-name-input").value = foodName;
  clearAutocomplete();
  analyzeByName();
}

function clearAutocomplete() {
  const dropdown = document.getElementById("autocomplete-dropdown");
  if (dropdown) {
    dropdown.remove();
  }
  document.removeEventListener("click", handleClickOutside);
}

function handleClickOutside(e) {
  const dropdown = document.getElementById("autocomplete-dropdown");
  const input = document.getElementById("food-name-input");

  if (dropdown && !dropdown.contains(e.target) && e.target !== input) {
    clearAutocomplete();
  }
}

// Quick search function for suggestions
async function quickSearch(foodName) {
  document.getElementById("search-input").value = foodName;
  await searchFoods();
}
