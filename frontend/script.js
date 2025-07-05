// API Configuration
const API_BASE_URL = "http://localhost:8000";

// Application State
const app = {
  currentTab: "name",
  loading: false,
  searchResults: [],
};

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

function initializeApp() {
  setupEventListeners();
  setupNavigation();
  checkAPIHealth();
}

function setupEventListeners() {
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
  document
    .getElementById("analyze-name-btn")
    .addEventListener("click", analyzeByName);
  document
    .getElementById("food-name-input")
    .addEventListener("keypress", (e) => {
      if (e.key === "Enter") analyzeByName();
    });
  document
    .getElementById("food-name-input")
    .addEventListener("input", handleFoodNameInput);

  // Image upload
  const uploadArea = document.getElementById("upload-area");
  const imageInput = document.getElementById("image-input");

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

  // Manual analysis
  document
    .getElementById("analyze-manual-btn")
    .addEventListener("click", analyzeManualData);

  // Search
  document.getElementById("search-btn").addEventListener("click", searchFoods);
  document.getElementById("search-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") searchFoods();
  });

  // Toast close
  document.querySelector(".toast-close").addEventListener("click", hideToast);
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
  // Update navigation
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle(
      "active",
      link.getAttribute("href") === `#${sectionName}`
    );
  });

  // Update sections
  document.querySelectorAll(".section").forEach((section) => {
    section.style.display = section.id === sectionName ? "block" : "none";
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

  showLoading(true);

  try {
    const response = await fetch(
      `${API_BASE_URL}/analyze/food/${encodeURIComponent(foodName)}`
    );
    const data = await response.json();

    if (response.ok) {
      displayResults(data.analysis);
      showSection("analyze");
      document.getElementById("results").style.display = "block";
    } else {
      showToast(data.detail || "Food not found in database");
    }
  } catch (error) {
    showToast("Error analyzing food: " + error.message);
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
      showSection("analyze");
      document.getElementById("results").style.display = "block";
    } else {
      showToast(data.detail || "Error analyzing nutrition data");
    }
  } catch (error) {
    showToast("Error analyzing nutrition data: " + error.message);
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
  // Food Info
  const foodInfoHtml = `
        <div class="food-info-item">
            <span class="food-info-label">Food Name</span>
            <span class="food-info-value">${analysis.name}</span>
        </div>
        <div class="food-info-item">
            <span class="food-info-label">Food Group</span>
            <span class="food-info-value">${analysis.food_group}</span>
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
  document.getElementById("food-info").innerHTML = foodInfoHtml;

  // Nutrition Facts
  const nutrition = analysis.nutrition;
  const nutritionHtml = `
        <div class="nutrition-grid">
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.calories}</span>
                <div class="nutrition-label">Calories</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.protein}g</span>
                <div class="nutrition-label">Protein</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.fat}g</span>
                <div class="nutrition-label">Fat</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.carbohydrates}g</span>
                <div class="nutrition-label">Carbs</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.fiber}g</span>
                <div class="nutrition-label">Fiber</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.sugar}g</span>
                <div class="nutrition-label">Sugar</div>
            </div>
            <div class="nutrition-item">
                <span class="nutrition-value">${nutrition.sodium}mg</span>
                <div class="nutrition-label">Sodium</div>
            </div>
        </div>
    `;
  document.getElementById("nutrition-facts").innerHTML = nutritionHtml;

  // Health Tags
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
  document.getElementById("health-tags").innerHTML = tagsHtml;

  // AI Insights
  let insightsHtml = "<p>AI insights are being generated...</p>";
  if (analysis.ai_insights) {
    insightsHtml = Object.entries(analysis.ai_insights)
      .map(
        ([key, value]) => `
                <div class="ai-insight">
                    <h4>${formatInsightTitle(key)}</h4>
                    <p>${value}</p>
                </div>
            `
      )
      .join("");
  }
  document.getElementById("ai-insights").innerHTML = insightsHtml;
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
  let insightsHtml = "<p>AI insights are being generated...</p>";
  if (
    data.ai_insights &&
    data.ai_insights.message !== "AI insights not available"
  ) {
    insightsHtml = Object.entries(data.ai_insights)
      .map(
        ([key, value]) => `
                <div class="ai-insight">
                    <h4>${formatInsightTitle(key)}</h4>
                    <p>${value}</p>
                </div>
            `
      )
      .join("");
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
  // You could implement autocomplete here by calling the search API
  // For now, we'll keep it simple
}
