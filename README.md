# Project idea

This algorithm is based on a real request from the software testing department for physical modeling. There is data on the automated test database, which contains approximately 1,700 tests. The tests are designed to catch problems in the code after developer input. The automated test database was created over the past 12 years, and there is no information on the quality of the tests (how well they are configured to catch errors).

Every night, all tests are run on the updated version of the product. The test results may reveal some tests that fail. Since the relationships between tests are complex and incomprehensible, the tester only checks for failures in individual tests (identified in the daily error report). However, experience shows that such a superficial analysis is insufficient. The question arises as to why similar tests (for the same physical process, added on the same day, by the same person, etc.) behave differently.

An idea emerged to create a smart analyzer—a clusterer—that would find explicit and implicit connections between test projects, something inaccessible to the human mind. The analyzer's output would group tests with similar parameters. Then, after receiving a crash report for one test, a tester could check the entire group of similar tests that did not report any errors.

As a result, the analyzer-clustering unit receives data from the test report about which tests failed after the merge, and the developed analyzer produces a list of similar tests that supposedly passed.

# Project composition

- Collecting_data: collecting data from the test report database into a single data frame
- EDA: analysis and description of the current automated test base
- Clustering_baseline: selection and testing of an algorithm for clustering
- SP_parameters_tuning : hyperparameter tuning
- Conclusion: final dataset and conclusions
