# Technical Requirements Checklist for Dev Team Approval  

To gain approval from the development team, ensure the computation module meets the following technical requirements.  

## **Successful Execution**  
- [ ] The module runs successfully with **three or more sites** on the public platform using the provided test data.
  - Needs testing by dev team

## **Computation Description Document**  
Provide a clear and comprehensive document covering the following:  
- [x] **Algorithm Description** – Explanation of the methodology used.  
- [x] **Limitations** – Any constraints or known issues with the algorithm.  
- [ ] **Input Data Specification**:  
   - [x] Structure of the **data directory**.  
   - [ ] Specification for **`parameters.json`**. 
     - Needs work. What values and specifications for the covariate types?
- [ ] **Output Format Description** – Clear definition of expected outputs.
  - Describe html and json outputs
  - Provide examples
- [ ] **Minimum Hardware & Space Requirements** – System requirements for execution.  
- [ ] **Basic Dataset Validator** – A tool or script to validate input data format.  

## **GitHub Repository**  
Ensure the module is properly hosted and documented:  
- [x] The module is in a **publicly accessible repository**.  
- [ ] The repository includes:  
   - [x] A **buildable, working image**.  
   - [ ] **Test data** for validation (**3 or more sites**).
     - Needs an additional site
   - [x] The **computation description document**.  
