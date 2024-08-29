let checkInterval = null;
let intervals = [];
const CHECK_INTERVAL_MS = 10000; // 10 seconds

function createIndicator() {
  const indicator = document.createElement("div");
  indicator.style.position = "absolute";
  indicator.style.right = "10px";
  indicator.style.top = "10px";
  indicator.style.width = "20px";
  indicator.style.height = "20px";
  indicator.style.borderRadius = "50%";
  indicator.style.border = "1px solid #000";
  indicator.style.backgroundColor = "green";
  indicator.className = "status-indicator";
  return indicator;
}

function createSpinner() {
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  spinner.style.position = "absolute";
  spinner.style.right = "40px";
  spinner.style.top = "10px";
  spinner.style.width = "20px";
  spinner.style.height = "20px";
  spinner.style.border = "2px solid #ccc";
  spinner.style.borderTop = "2px solid #000";
  spinner.style.borderRadius = "50%";
  spinner.style.animation = "spin 1s linear infinite";
  spinner.style.display = "none";
  return spinner;
}

// Add spinner animation
const style = document.createElement('style');
style.innerHTML = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);

// Function to check the description and update the indicator
async function checkDescription(descriptionField, indicator, spinner) {
  const description = descriptionField.value;
  spinner.style.display = "block"; // Show spinner
  const containsSecret = await checkForSecrets(description);
  spinner.style.display = "none"; // Hide spinner

  if (containsSecret.prediction) {
    indicator.style.backgroundColor = "red";
    results = "Contains strings that might be secret"
    for (let i = 0; i < containsSecret.candidates.length; i++) {
      results+= "\n";
      results+= (i+1).toString()
      results+=". "
      results+= containsSecret.candidates[i]
    }
    indicator.title = results;
    } else {
    indicator.style.backgroundColor = "green";
    indicator.title = "You're safe";
  }
}

function startChecking(descriptionField) {
  const indicator = createIndicator();
  const spinner = createSpinner();
  descriptionField.parentElement.style.position = "relative"; // Ensure parent has position
  descriptionField.parentElement.appendChild(indicator);
  descriptionField.parentElement.appendChild(spinner);
  console.log("Checking description...");
  checkInterval = setInterval(() => {
    console.log("Checking description2...");
    checkDescription(descriptionField, indicator, spinner);
  }, CHECK_INTERVAL_MS);
  intervals.push(checkInterval);

  // if intervals size more than 10000, then start removing the first element
  if (intervals.length > 10000) {
    clearInterval(intervals.shift());
  }
}

async function checkForSecrets(description) {
  try {
    console.log(description);
    const response = await fetch("https://103.94.135.163:5000/checkdescription", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ description }),
    });

    const result = await response.json();
    console.log("Result from backend:", result); // Debugging line
    return result;
  } catch (error) {
    console.error("Error checking for secrets:", error);
    return {
      prediction: false,
      candidates: []
    }
  }
}

let inside = false;


function handleNewIssuePage() {
  const descriptionField = document.querySelector("#issue_body");
  console.log(descriptionField);

  if (descriptionField) {
    startChecking(descriptionField);
  } else {
    console.log("descriptionField not found, setting up observer...");
    const observer = new MutationObserver((mutationsList, observer) => {
      for (let mutation of mutationsList) {
        if (mutation.type === 'childList') {
          const newDescriptionField = document.querySelector("#issue_body");
          if (newDescriptionField) {
            observer.disconnect();
            startChecking(newDescriptionField);
            break;
          }
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
}

function addEventListener(event) {
  console.log(checkInterval);
  intervals.forEach((interval) => {
    clearInterval(interval);
  });
  intervals = [];
  console.log("baire dhukse");
  const pathname = new URL(event.destination.url).pathname;
  console.log(pathname);
  const pathnamePattern = /\/*\/issues\/new(\/)?/;

  if (pathnamePattern.test(pathname)) {
    if (inside) {
      console.log("if vitore dhukse");
      inside = false;
      intervals.forEach((interval) => {
        clearInterval(interval);
      });
      intervals = [];
    
      handleNewIssuePage();
    } else {
      console.log("else vitore dhukse");

      
      handleNewIssuePage();
      inside = true;
    }
  
  }
}

function init() {
  navigation.addEventListener("navigate", (event) => {
    addEventListener(event);
  });
 
}

init();
