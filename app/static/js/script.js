function openMenu() {
    document.getElementById("menu").style.width = "250px";
}

function closeMenu() {
    document.getElementById("menu").style.width = "0";
}

//##############################################################################################

// Function to hide the loading message
function hide_word() {
    const loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'none';
}

//##############################################################################################
// live detecting

// function updateDetectionStatus(detectedItems) {
//     const items = document.querySelectorAll('#detection-list li');

//     items.forEach(item => {
//         const itemName = item.getAttribute('data-item');
//         if (detectedItems.includes(itemName)) {
//             item.classList.remove('not-detected');
//             item.classList.add('detected');
//         } else {
//             item.classList.remove('detected');
//             item.classList.add('not-detected');
//         }
//     });
// }

// setInterval(() => {
//     fetch('/live-detect')
//         .then(response => response.json())
//         .then(data => {
//             const detectedItems = data.detected_items;
//             updateDetectionStatus(detectedItems);
//         })
//         .catch(error => console.error('Error fetching detection data:', error));
// }, 5000);

//##############################################################################################
// image detecting

//##############################################################################################

