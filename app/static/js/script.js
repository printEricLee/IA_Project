// function openMenu() {
//     document.getElementById("menu").style.width = "250px";
// }

// function closeMenu() {
//     document.getElementById("menu").style.width = "0";
// }

// Function to hide the loading message
function hide_word() {
    const loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'none';
}

// Function to show the item boxes after loading
async function show_boxes() {
    const itemList = document.getElementById('itemList');
    itemList.style.display = 'block'; // Make the list visible
    console.log("Item list is now visible."); // Debug log

    // Fetch detected items
    const detectedItems = await getDetectedItems();
    
    // Log detected items
    console.log("Detected items:", detectedItems);

    // Check if there are any detected items
    if (!Array.isArray(detectedItems) || detectedItems.length === 0) {
        console.log("No items detected."); // Debug log for no detected items
        itemList.innerHTML = "<div class='item not-detected'>No items detected</div>";
    } else {
        console.log("Detected items array is valid."); // Log valid array check
        updateItemColors(detectedItems);
    }
}

//##############################################################################################

// Function to fetch detected items from the server
async function getDetectedItems() {
    try {
        const response = await fetch('/api/get-detected-items'); // Adjust this endpoint as necessary
        if (!response.ok) {
            console.error('Network response was not ok:', response.statusText);
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        console.log("Fetched detected items:", data); // Debug log for fetched data
        
        // Access the detected_items property
        return data.detected_items; // Return the detected items array
    } catch (error) {
        console.error("Error fetching detected items:", error);
        return []; // Return an empty array in case of error
    }
}

// Function to update the colors of the item boxes based on detection
function updateItemColors(detectedItems) {
    const items = document.querySelectorAll('#itemList .item');

    items.forEach(item => {
        const itemId = item.id.trim(); // Trim whitespace for safety
        console.log(`Checking item: ${itemId}`); // Debug log

        if (detectedItems.includes(itemId)) {
            console.log(`Detected: ${itemId}`); // Debug log for detected items
            item.classList.remove('not-detected');
            item.classList.add('detected');
            item.style.backgroundColor = 'green'; // Change color for detected items
        } else {
            console.log(`Not Detected: ${itemId}`); // Debug log for not detected items
            item.classList.remove('detected');
            item.classList.add('not-detected');
            item.style.backgroundColor = 'grey'; // Reset color for not detected items
        }
    });
}

// Mock function for testing (uncomment to use during development)
async function getDetectedItems() {
    // Mock response for testing
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(["dirts"]); // Simulate detecting "dirts"
        }, 1000);
    });
}

//##############################################################################################
