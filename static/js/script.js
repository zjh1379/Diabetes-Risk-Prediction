// Wait for document to fully load
document.addEventListener('DOMContentLoaded', function() {
    // Get current language from HTML document
    const currentLang = document.documentElement.lang;
    
    // Translation dictionary
    const translations = {
        'zh-CN': {
            'requiredField': '此字段必填',
            'minValue': '最小值为',
            'maxValue': '最大值为',
            'processing': '处理中...',
            'underweight': '偏瘦',
            'normal': '正常',
            'overweight': '超重',
            'obese': '肥胖'
        },
        'en': {
            'requiredField': 'This field is required',
            'minValue': 'Minimum value is',
            'maxValue': 'Maximum value is',
            'processing': 'Processing...',
            'underweight': 'Underweight',
            'normal': 'Normal',
            'overweight': 'Overweight',
            'obese': 'Obese'
        }
    };
    
    // Get translation based on current language
    function getTranslation(key) {
        const lang = currentLang || 'en';
        return (translations[lang] && translations[lang][key]) || translations['en'][key];
    }
    
    // Get form elements
    const tabButtons = document.querySelectorAll('.nav-link');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const nextButtons = document.querySelectorAll('.next-tab');
    const prevButtons = document.querySelectorAll('.prev-tab');
    const form = document.getElementById('assessment-form');

    // Form validation function
    function validateTab(tabPane) {
        let isValid = true;
        const requiredInputs = tabPane.querySelectorAll('input[required], select[required]');
        
        requiredInputs.forEach(input => {
            if (input.value.trim() === '') {
                isValid = false;
                input.classList.add('is-invalid');
                
                // Add error feedback if not present
                if (!input.nextElementSibling || !input.nextElementSibling.classList.contains('invalid-feedback')) {
                    const feedback = document.createElement('div');
                    feedback.className = 'invalid-feedback';
                    feedback.textContent = getTranslation('requiredField');
                    input.parentNode.insertBefore(feedback, input.nextElementSibling);
                }
            } else {
                input.classList.remove('is-invalid');
                // Remove any existing error feedback
                if (input.nextElementSibling && input.nextElementSibling.classList.contains('invalid-feedback')) {
                    input.parentNode.removeChild(input.nextElementSibling);
                }
            }
        });
        
        return isValid;
    }

    // Add event listeners to Next buttons
    nextButtons.forEach(button => {
        button.addEventListener('click', function() {
            const activeTabPane = document.querySelector('.tab-pane.active');
            if (validateTab(activeTabPane)) {
                // Find current tab index
                let currentIndex = -1;
                for (let i = 0; i < tabPanes.length; i++) {
                    if (tabPanes[i] === activeTabPane) {
                        currentIndex = i;
                        break;
                    }
                }
                
                // Activate next tab if not the last one
                if (currentIndex < tabPanes.length - 1) {
                    const nextTab = new bootstrap.Tab(tabButtons[currentIndex + 1]);
                    nextTab.show();
                }
            }
        });
    });

    // Add event listeners to Previous buttons
    prevButtons.forEach(button => {
        button.addEventListener('click', function() {
            const activeTabPane = document.querySelector('.tab-pane.active');
            // Find current tab index
            let currentIndex = -1;
            for (let i = 0; i < tabPanes.length; i++) {
                if (tabPanes[i] === activeTabPane) {
                    currentIndex = i;
                    break;
                }
            }
            
            // Activate previous tab if not the first one
            if (currentIndex > 0) {
                const prevTab = new bootstrap.Tab(tabButtons[currentIndex - 1]);
                prevTab.show();
            }
        });
    });

    // Form submission validation
    if (form) {
        form.addEventListener('submit', function(event) {
            const activeTabPane = document.querySelector('.tab-pane.active');
            if (!validateTab(activeTabPane)) {
                event.preventDefault();
                return false;
            }
            
            // Add loading indicator on submit
            const submitButton = document.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ' + getTranslation('processing');
            }
            
            return true;
        });
    }

    // Number input validation
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));
            const value = parseFloat(this.value);
            
            if (value < min) {
                this.setCustomValidity(`${getTranslation('minValue')} ${min}`);
            } else if (value > max) {
                this.setCustomValidity(`${getTranslation('maxValue')} ${max}`);
            } else {
                this.setCustomValidity('');
            }
        });
    });

    // BMI automatic calculation
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    
    function calculateBMI() {
        if (heightInput && weightInput && 
            heightInput.value.trim() !== '' && weightInput.value.trim() !== '') {
            const height = parseFloat(heightInput.value) / 100; // Convert to meters
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                const bmiElement = document.getElementById('bmi-value');
                if (bmiElement) {
                    bmiElement.textContent = bmi.toFixed(1);
                    
                    // Update BMI status
                    const bmiStatus = document.getElementById('bmi-status');
                    if (bmiStatus) {
                        if (bmi < 18.5) {
                            bmiStatus.textContent = getTranslation('underweight');
                            bmiStatus.className = 'text-info';
                        } else if (bmi < 24) {
                            bmiStatus.textContent = getTranslation('normal');
                            bmiStatus.className = 'text-success';
                        } else if (bmi < 28) {
                            bmiStatus.textContent = getTranslation('overweight');
                            bmiStatus.className = 'text-warning';
                        } else {
                            bmiStatus.textContent = getTranslation('obese');
                            bmiStatus.className = 'text-danger';
                        }
                    }
                }
            }
        }
    }
    
    // Add event listeners for height and weight inputs
    if (heightInput && weightInput) {
        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }
}); 