#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
 
int main()
{
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 
    std::random_device rd;
    std::mt19937_64 g(0);
 
    std::shuffle(v.begin(), v.end(), g);
 
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";
}
