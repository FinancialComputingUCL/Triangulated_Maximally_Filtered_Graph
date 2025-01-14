# Original TMFG Implementation (Replaced in 2025/1)
Although the underlying TMFG algorithm was already efficient, we have further optimised the code implementation to improve its performance. The new implementation is significantly faster (demonstrated [here](plot.ipynb)), especially for large datasets, and provides a more scalable solution. 
## Implementation of the new optimisation algorithm
- Perform operations mainly in NumPy, leveraging its efficient vectorised computations.
- Use a boolean NumPy array to track remaining vertices for insertion, instead of using `np.setdiff1d` everytime while finding reaming vertices. This enables efficient identification of the best remaining vertex for triangles.
- Implement a cache to sum columns corresponding to triangles while finding the best gain, optimising the most time-consuming and repetitive operation.
- Utilise a priority queue (implemented with `heapq`) to find the pre-calculated vertex and triangle pair in each insertion iteration, reducing search complexity from O(n) to O(log n).
- Refactor code and add docstrings for improved readability and maintainability.