"""Day 10 AoC 2025 solver."""

from itertools import combinations, chain
from ast import literal_eval
from scipy.optimize import linprog
import numpy as np


class Solver10():
    """Provide solution to Day 10.

    Filename of text file to be passed as argument on initialisation.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])

        # manual items
        self.indicator_lights: np.array = np.array([])
        self.buttons: np.array = np.array([])
        self.joltage_levels: np.array = np.array([])

        # solutions dict
        self.solutions: dict = {"Part One": 0, "Part Two": 0}

        # import data
        self.import_data(filename)

    def __call__(self):
        """Full solve."""
        self.solve_part_one()
        self.solve_part_two()

# =============================================================================
# ===== DATA INPUT AND CONVERSION =====
# =============================================================================

    def import_data(self, filename: str):
        """Import supplied data from text file."""
        self.data = np.loadtxt(
            fname=filename,
            dtype='O',
            comments=None,
            delimiter='@'  # not in file, therefore imports whole lines
            )

        # ===== INDICATOR LIGHT DIAGRAMS =====

        # extract indicator light diagrams data
        self.indicator_lights = [x.split()[0][1:-1] for x in self.data]

        # convert to int/boolean form
        self.indicator_lights = [
                np.fromiter((x == '#' for x in y), int)
                for y in self.indicator_lights
                ]

        # ===== BUTTON WIRING SCHEMATICS =====

        # extract button wiring schematics
        self.buttons = [x.split()[1:-1] for x in self.data]

        # convert to array form
        self.buttons = [
            [np.array(list(literal_eval(x)))
             if hasattr(literal_eval(x), "__iter__") else
             np.array([literal_eval(x)])
             for x in y]
            for y in self.buttons
            ]

        # ===== JOLTAGE REQUIREMENTS =====

        # extract joltage levels
        self.joltage_levels = [x.split()[-1][1:-1] for x in self.data]

        # convert to array form
        self.joltage_levels = [
            np.array(list(literal_eval(x))) for x in self.joltage_levels
            ]

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def difference_matrix(self, a: np.array, b: np.array) -> np.array:
        """
        Calculate all combinations of first differences of rows in a linear \
            system.

        Given 2d matrix a and column vector b, which form a linear system, \
        calculate all combinations of first differences, returned as a new \
        linear system.
        """
        # ===== ERROR CHECKING =====

        # error check for matrix / vector dimension compatibility
        if a.shape[0] != b.shape[0]:
            return False

        # ===== INITIALISE VARIABLES =====

        # form iterable for row combinations
        row_iter = list(combinations(range(a.shape[0]), r=2))

        # number of combinations (choose 2) from row length of array a
        iter_len = int(len(row_iter))

        # initialise working variables for differences
        row_diff = np.array([])
        val_diff: int = 0

        # initialise output difference matrices
        a_diff = np.zeros(
            (iter_len, a.shape[1]),
            dtype=int
            )
        b_diff = np.zeros(
            (iter_len,),
            dtype=int
            )

        # ===== FIRST DIFFERENCES =====

        # iterate through unique row combinations
        for row_idx, row_combi in enumerate(row_iter):

            # take row differences to solve for button press variables
            row_diff = a[row_combi[0]] - a[row_combi[1]]
            val_diff = b[row_combi[0]] - b[row_combi[1]]

            # fill difference matrix
            a_diff[row_idx] = row_diff * np.where(val_diff < 0, -1, 1)
            b_diff[row_idx] = val_diff * np.where(val_diff < 0, -1, 1)

        return a_diff, b_diff

    def n_addition_matrix(self, a: np.array, b: np.array, n=None) -> np.array:
        """
        Calculate all combinations of n additions of rows in a linear system.

        Given 2d matrix a and column vector b, which form a linear system, \
        calculate all combinations of n additions, returned as a new \
        linear system.

        If n not specified, all addtitions up to the full number of rows will \
        be calculated.
        """
        # ===== ERROR CHECKING =====

        # error check for matrix / vector dimension compatibility
        if a.shape[0] != b.shape[0]:
            return False

        # ===== INITIALISE VARIABLES =====

        # form iterable for row combinations
        row_iter: list = None  # list(combinations(range(a.shape[0]), r=n))

        # number of combinations (choose n) from row length of array a
        iter_len: int = 0

        # initialise working variables for additions
        row_sum = np.array([])
        val_sum: int = 0

        # initialise working addition matrices
        a_sum, b_sum = (np.array([]),) * 2

        # initialise output matrix / vector
        output_matrix, output_vector = (np.array([]),) * 2

        # ===== NTH SUMMATIONS =====

        # between 2 and n (number of matrix rows) additions, unless explicitly
        # overidden in input argument n
        for number in range(2, (a.shape[0] if n is None else n)+1):

            # generate addition combinations
            row_iter = list(combinations(range(a.shape[0]), r=number))

            # number of combinations (choose n) from row length of array a
            iter_len = int(len(row_iter))

            # addition matrices
            a_sum = np.zeros((iter_len, a.shape[1]), dtype=int)
            b_sum = np.zeros((iter_len,), dtype=int)

            # iterate through unique row combinations
            for row_idx, row_combi in enumerate(row_iter):

                # take row additions to solve for button press variables
                row_sum = np.sum(a[[*row_combi]], axis=0)
                val_sum = np.sum(b[[*row_combi]], axis=0)

                # fill addition matrix
                a_sum[row_idx] = row_sum * np.where(val_sum < 0, -1, 1)
                b_sum[row_idx] = val_sum * np.where(val_sum < 0, -1, 1)

            # append to output
            if not np.any(output_matrix):
                output_matrix = a_sum.copy()
                output_vector = b_sum.copy()
            else:
                output_matrix = np.concatenate([output_matrix, a_sum], axis=0)
                output_vector = np.concatenate([output_vector, b_sum], axis=0)

        return output_matrix, output_vector

    def unique_solutions(self, a: np.array, b: np.array) -> np.array:
        """Evaluate linear system for unique solutions by iterative back- \
        substitution.

        The linear system is traversed for common elements and reduced \
        iteratively by substitution, until unique components of the system \
        remain.
        """
        # ===== ERROR CHECKING =====

        # error check for matrix / vector dimension compatibility
        if a.shape[0] != b.shape[0]:
            return False

        # ===== INITIALISE VARIABLES =====

        # intialise matrix and vector for comparison and reduction
        a_matrix_1, a_matrix_2 = (a.copy(),) * 2
        b_vector_1, b_vector_2 = (b.copy(),) * 2

        # working variable for non-zero elements in each row
        non_zero_cols: np.array = np.array([])

        # working variable for matching rows
        match: np.array = np.array([])

        # initialise loop break condition
        loop_break: bool = False

        # ===== TRAVERSE LINEAR SYSTEM =====

        while True:

            # set traversal to run once, unless solution found
            loop_break = True

            # search through rows, checking for common element(s) and \
            # subsequently eliminating from the linear system when found
            for idx, row in enumerate(a_matrix_1):

                # skip null row
                if not np.any(row):
                    continue

                # log non-zero column indices
                non_zero_cols = row.nonzero()[0]

                # match row from 1st matrix to any row in 2nd matrix
                match = np.all(np.equal(row[non_zero_cols],
                                        a_matrix_2[:, non_zero_cols]), axis=1)

                # exclude current index in 2nd matrix
                match[idx] = np.False_

                # eliminate solutions from 2nd matrix / vector
                if np.any(match):

                    # eliminate solutions from 2nd matrix
                    a_matrix_2[np.ix_(match, non_zero_cols)] = 0

                    # reduce column vector value by solution
                    b_vector_2[match] -= b_vector_1[idx]

                    # restart traversal after new solution found
                    loop_break = False

            # roll matrix / vector 1 to latest state
            a_matrix_1 = a_matrix_2.copy()
            b_vector_1 = b_vector_2.copy()

            # check for non-continuation of loop
            if loop_break:

                # no solutions found after full traversal
                break

        # ===== SANITISE OUTPUT =====

        # if solution value is negative, convert to
        # positive analogue in ouput matrix / vector
        for row in range(a_matrix_1.shape[0]):

            if np.sign(b_vector_1[row]) == -1:
                a_matrix_1[row] *= -1
                b_vector_1[row] *= -1

        # filter null rows from solution linear system
        b_vector_1 = b_vector_1[np.count_nonzero(a_matrix_1, axis=1) != 0]
        a_matrix_1 = a_matrix_1[np.count_nonzero(a_matrix_1, axis=1) != 0]

        return a_matrix_1, b_vector_1

    def update_solution_vector(self,
                               a: np.array,
                               b: np.array,
                               x: np.array
                               ) -> np.array:
        """Update solution ranges of a linear system of the form a @ x = b.

        Given 2d matrix a and column vector b, which form a linear system, as \
        well as a solution column vector x, update the ranges of solutions to \
        the linear system.

        x is a structured numpy array with 3 fields, with datatypes:

            dtype=[('floor', int), ('ceiling', int), ('solution', int)]

        Initial integer values for 'floor' & 'ceiling' are to be supplied \
        according to the constraints of the each solution variable, whilst \
        a default 'solution' of -1 can be supplied; this will be returned if \
        a solution is not found.

        The range of values (bounded by floor / ceiling) of a each variable in
        a linear system are updated based on the expressed relationships.
        When floor value and ceiling values converge, the final value is
        returned.
        """
        # ===== ERROR CHECKING =====

        # error check for matrix / vector dimension compatibility
        if a.shape[0] != b.shape[0]:
            return False

        # ===== INITIALISE VARIABLES =====

        # output vector for updated bounds
        x_output = x.copy()

        # working variable for array of non-zero column indices
        non_zero_cols: np.array = np.array([])

        # working variable for bounds
        bounds: dict = {
            "floor": np.array(0, dtype=[("sum", int), ("headroom", int)]),
            "ceiling": np.array(0, dtype=[("sum", int), ("headroom", int)])
            }

        # loop condition
        loop_break = False

        # working variable for difference equations
        bifur_1: np.array = np.array([])
        bifur_2: np.array = np.array([])

        # ===== TRAVERSE LINEAR SYSTEM =====

        while True:

            # set traversal to run once, unless update made to solution vector
            loop_break = True

            # traverse each row of matrix
            for idx, row in enumerate(a):

                # log indices of non-zero elements per row
                non_zero_cols = row.nonzero()[0]

                # ===== ADDITIVE EQUATIONS =====

                # all positive elements
                if np.all(np.sign(row[non_zero_cols]) == 1):

                    # traverse each non-zero element per row
                    for idx_nz in non_zero_cols:

                        # ===== UPDATE CEILINGS =====

                        # calculate sum of floor values of other elements
                        bounds['floor']['sum'] = np.sum(
                            x_output[
                                non_zero_cols[non_zero_cols != idx_nz]
                                ]['floor'] *
                            row[non_zero_cols[non_zero_cols != idx_nz]])

                        # derive ceiling headroom, floored at 0
                        bounds['ceiling']['headroom'] = max(
                            b[idx] - bounds['floor']['sum'], 0)

                        # check if element ceiling exceeds ceiling headroom

                        # loop condition updated if so
                        loop_break = not np.sign(
                            x_output[idx_nz]['ceiling'] -
                            bounds['ceiling']['headroom'] // row[idx_nz]) == 1

                        # cap element ceiling amount iff > ceiling headroom
                        x_output[idx_nz]['ceiling'] = \
                            x_output[idx_nz]['ceiling'] if loop_break else \
                            bounds['ceiling']['headroom'] // row[idx_nz]

                        # ===== UPDATE FLOORS =====

                        # calculate sum of ceiling values of other elements
                        bounds['ceiling']['sum'] = np.sum(
                            x_output[
                                non_zero_cols[non_zero_cols != idx_nz]
                                ]['ceiling'] *
                            row[non_zero_cols[non_zero_cols != idx_nz]]
                            )

                        # derive floor headroom, floored at 0
                        bounds['floor']['headroom'] = max(
                            b[idx] - bounds['ceiling']['sum'], 0)

                        # check if element floor is less than floor headroom

                        # loop condition updated if so
                        loop_break = not np.sign(
                            bounds['floor']['headroom'] // row[idx_nz] -
                            x_output[idx_nz]['floor']) == 1

                        # increase element floor amount iff < floor headroom
                        x_output[idx_nz]['floor'] = \
                            x_output[idx_nz]['floor'] if loop_break else \
                            bounds['floor']['headroom'] // row[idx_nz]

                # ===== DIFFERENCE EQUATIONS =====

                else:

                    # bifurcate difference into component additive equations

                    # half 1: all positive elements
                    bifur_1 = non_zero_cols[np.sign(row[non_zero_cols]) == 1]
                    # half 2: all negative elements
                    bifur_2 = non_zero_cols[np.sign(row[non_zero_cols]) == -1]

                    # ===== FIRST HALF =====

                    # traverse each non-zero positive element
                    for idx_nz in bifur_1:

                        # calculate sum of ceiling values of other elements
                        bounds['ceiling']['sum'] = np.sum(
                            x_output[
                                bifur_1[bifur_1 != idx_nz]]['ceiling'] *
                            row[bifur_1[bifur_1 != idx_nz]])

                        # calculate sum of floor values of other elements
                        bounds['floor']['sum'] = np.sum(
                            x_output[
                                bifur_1[bifur_1 != idx_nz]]['floor'] *
                            row[bifur_1[bifur_1 != idx_nz]])

                        # ===== UPDATE CEILINGS =====

                        # update each ceiling, based on ceiling values of other
                        # elements

                        # derive ceiling headroom
                        bounds['ceiling']['headroom'] = b[idx] + np.sum(
                            x_output[bifur_2]['ceiling'] * -row[bifur_2])

                        # check if element ceiling exceeds ceiling headroom

                        # loop condition updated if so
                        loop_break = not np.sign(
                            x_output[idx_nz]['ceiling'] -
                            bounds['ceiling']['headroom'] // row[idx_nz]) == 1

                        # cap element ceiling amount iff > ceiling headroom
                        x_output[idx_nz]['ceiling'] = \
                            x_output[idx_nz]['ceiling'] if loop_break else \
                            bounds['ceiling']['headroom'] // row[idx_nz]

                        # check if element ceiling exceeds ceiling headroom
                        # minus floor values of other elements. Rationale is
                        # ceiling headroom minus floor sum is maximum ceiling
                        # value

                        # loop condition updated if so
                        loop_break = not np.sign(
                            x_output[idx_nz]['ceiling'] -
                            (bounds['ceiling']['headroom'] -
                             bounds['floor']['sum']) // row[idx_nz]) == 1

                        # cap element ceiling amount iff > ceiling headroom
                        # minus floor sum
                        x_output[idx_nz]['ceiling'] = \
                            x_output[idx_nz]['ceiling'] if loop_break else \
                            (bounds['ceiling']['headroom'] -
                             bounds['floor']['sum']) // row[idx_nz]

                        # ===== UPDATE FLOORS =====

                        # update each floor, based on ceiling values of other
                        # elements

                        # derive floor headroom
                        bounds['floor']['headroom'] = b[idx] + np.sum(
                            x_output[bifur_2]['floor'] * -row[bifur_2])

                        # check if element floor is less tan floor headroom
                        # minus ceiling values of other elements. Rationale is
                        # floor headroom minus ceiling sum is minimum floor
                        # value

                        # loop condition updated if so
                        loop_break = not np.sign(
                            (bounds['floor']['headroom'] -
                             bounds['ceiling']['sum']) // row[idx_nz] -
                            x_output[idx_nz]['floor']) == 1

                        # increase element ceiling amount iff < floor headroom
                        # minus ceiling sum
                        x_output[idx_nz]['floor'] = \
                            x_output[idx_nz]['floor'] if loop_break else \
                            (bounds['floor']['headroom'] -
                             bounds['ceiling']['sum']) // row[idx_nz]

                    # ===== SECOND HALF =====

                    # traverse each non-zero negative element
                    for idx_nz in bifur_2:

                        # calculate sum of ceiling values of other elements
                        bounds['ceiling']['sum'] = np.sum(
                            x_output[
                                bifur_2[bifur_2 != idx_nz]]['ceiling'] *
                            -row[bifur_2[bifur_2 != idx_nz]])

                        # calculate sum of floor values of other elements
                        bounds['floor']['sum'] = np.sum(
                            x_output[
                                bifur_2[bifur_2 != idx_nz]]['floor'] *
                            -row[bifur_2[bifur_2 != idx_nz]])

                        # ===== UPDATE CEILINGS =====

                        # update each ceiling, based on ceiling values of other
                        # elements

                        # derive ceiling headroom
                        bounds['ceiling']['headroom'] = np.sum(
                            x_output[bifur_1]['ceiling'] * row[bifur_1]) - \
                            b[idx]

                        # check if element ceiling exceeds ceiling headroom

                        # loop condition updated if so
                        loop_break = not np.sign(
                            x_output[idx_nz]['ceiling'] -
                            bounds['ceiling']['headroom'] // -row[idx_nz]) == 1

                        # cap element ceiling amount iff > ceiling headroom
                        x_output[idx_nz]['ceiling'] = \
                            x_output[idx_nz]['ceiling'] if loop_break else \
                            bounds['ceiling']['headroom'] // -row[idx_nz]

                        # check if element ceiling exceeds ceiling headroom
                        # minus floor values of other elements. Rationale is
                        # ceiling headroom minus floor sum is maximum ceiling
                        # value

                        # loop condition updated if so
                        loop_break = not np.sign(
                            x_output[idx_nz]['ceiling'] -
                            (bounds['ceiling']['headroom'] -
                             bounds['floor']['sum']) // -row[idx_nz]) == 1

                        # cap element ceiling amount iff > ceiling headroom
                        # minus floor sum
                        x_output[idx_nz]['ceiling'] = \
                            x_output[idx_nz]['ceiling'] if loop_break else \
                            (bounds['ceiling']['headroom'] -
                             bounds['floor']['sum']) // -row[idx_nz]

                        # ===== UPDATE FLOORS =====

                        # update each floor, based on ceiling values of other
                        # elements

                        # derive floor headroom
                        bounds['floor']['headroom'] = np.sum(
                            x_output[bifur_1]['floor'] * row[bifur_1]) - b[idx]

                        # check if element floor is less tan floor headroom
                        # minus ceiling values of other elements. Rationale is
                        # floor headroom minus ceiling sum is minimum floor
                        # value

                        # loop condition updated if so
                        loop_break = not np.sign(
                            (bounds['floor']['headroom'] -
                             bounds['ceiling']['sum']) // -row[idx_nz] -
                            x_output[idx_nz]['floor']) == 1

                        # increase element ceiling amount iff < floor headroom
                        # minus ceiling sum
                        x_output[idx_nz]['floor'] = \
                            x_output[idx_nz]['floor'] if loop_break else \
                            (bounds['floor']['headroom'] -
                             bounds['ceiling']['sum']) // -row[idx_nz]

            if loop_break:
                break

        # update solution vector value iff floor and ceiling have converged
        x_output['solution'] = np.where(
            x_output['floor'] == x_output['ceiling'],
            x_output['ceiling'], -1)

        return x_output

# =============================================================================
# ===== FEWEST PRESSES TO CONFIGURE INDICATOR LIGHTS =====
# =============================================================================

    def configure_indicator_lights(self,
                                   indicator_lights: list,
                                   buttons: list
                                   ) -> int:
        """Calculate fewest button presses to configure indicator lights, \
            given indicator light diagrams and button wiring schematics.

        Combinatoric approach adopted as scale of problem was seen to be of \
        contained magnitude; average 2^(~7) ~ 128 indicator light \
        possibilities, 179 indicator lights, early termination on solution.
        """
        # ===== INITIALISE VARIABLES =====

        # array of button presses
        button_presses: np.array = np.empty(
            shape=len(indicator_lights),
            dtype=int
            )

        button_combinations: np.array = np.empty(
            shape=len(indicator_lights),
            dtype='O'
            )

        # initialise number buttons to press
        number_of_buttons: int = 0

        # working array
        test_array: np.array = np.array([])

        # initialise loop condition
        loop_cond: bool = False

        # ===== TRAVERSE INDICATOR LIGHT DIAGRAMS =====

        for idx, ind_light in enumerate(indicator_lights):

            # number of button presses to maximum one press each
            while number_of_buttons <= len(buttons[idx]):

                # test loop condition
                if loop_cond:

                    # next indicator light diagram
                    break

                # initialise test array to show outcome of button presses
                test_array = np.zeros(ind_light.shape[0], dtype=int)

                # increment number of buttons to press
                number_of_buttons += 1

                # combinatoric iterator, ensuring each button is pressed once
                # in every combination (no repeats)
                button_iter = list(
                    combinations(buttons[idx],
                                 r=number_of_buttons)
                    )

                # ===== PRESS BUTTONS =====

                for button_combi in button_iter:

                    # "chain" flattens wiring from button combination into
                    # 1d list
                    for i in list(chain.from_iterable(button_combi)):

                        # toggle if pressed
                        test_array[i] = not test_array[i]

                    # if the combination matches the light diagram...
                    if np.all(test_array == ind_light):

                        # ...log number of presses...
                        button_presses[idx] = number_of_buttons

                        # ...and correct combination...
                        button_combinations[idx] = button_combi

                        # exit loop - next indicator light diagram
                        loop_cond = True
                        break

                    # reset test array if button combi not valid
                    test_array = np.zeros(ind_light.shape[0], dtype=int)

            # reset number of buttons to press
            number_of_buttons = 0

            # reset loop condition
            loop_cond = False

        return button_presses, button_combinations

# =============================================================================
# ===== FEWEST PRESSES TO CONFIGURE JOLTAGE LEVEL COUNTERS =====
# =============================================================================

    def configure_joltage_levels(self,
                                 joltage_levels: list,
                                 buttons: list,
                                 ) -> int:
        """Calculate fewest button presses to configure joltage level \
        counters, given joltage requirements and button wiring schematics.

        A linear algebra approach has been adopted. It is assumed that there \
        exists a linear combination of button presses that equates to the \
        required joltage levels, and that a linear system of equations can be \
        formed and solved in the form:

            a @ x = b

        Instead of relying on Gaussian elimination techniques for matrix \
        inversion - which is not always possible with non-square matrices - \
        a linear programming approach is adopted in addition to narrowing \
        solution possibilities by use of updating bounds.

        Ultimately, the crux of this problem is solving over-, well-, and \
        under-determined linear systems, with minimum distrete integers as a \
        solution constraint.

        The final approach is a combination of:
            1) trivial operations on the system revealing the solution: sums, \
            differences, reductions
            2) updating the solution vector bounds to converge to a solution
            3) use of HiGHS MIP solver (via SciPy wrapper), alongside an \
                objective function and constraints to solve the system.

        https://www.datasciencebase.com/intermediate/linear-algebra/linear-programming-introduction
        https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog
        https://ergo-code.github.io/HiGHS/stable/solvers/#MIP
        """
        # ===== INITIALISE VARIABLES =====

        # total button presses across all machines
        button_press_total: int = 0

        # working variable for checking for all-ones rows (solution row)
        all_ones: np.array = np.array([])

        # loop through each machine to configure the joltage levels
        for idx, joltage_array in enumerate(joltage_levels):

            # ===== INITIALISE VARIABLES =====

            # initialise matrix of buttons that affect each joltage level
            # rows represent each joltage level counter, and columns, each \
            # button
            button_matrix = np.array(
                [np.bincount(i, minlength=joltage_array.size)
                 for i in buttons[idx]]
                ).T

            # initialise column vector of joltage levels
            joltage_vector = joltage_array.copy()

            # establish a solution vector that enables a constraint-led
            # approach to machine button presses. This contains a floor/ \
            # ceiling/solution value for each machine button press count
            press_count_vector = np.array(
                list(
                    zip(
                        [0] * button_matrix.shape[1],  # 0 for lower bound
                        [
                            joltage_vector[np.where(button_matrix[:, i])].min()
                            for i in range(button_matrix.shape[1])
                            ],  # lowest joltage level affected by button for \
                                # upper bound
                        [-1] * button_matrix.shape[1],  # -1 default solution
                        )
                    ),
                dtype=[('floor', int), ('ceiling', int), ('solution', int)]
                )

            # initialise working variables for each component
            updated_matrix = button_matrix.copy()
            updated_vector = joltage_vector.copy()
            updated_press_count = press_count_vector.copy()

            # ===== INITIAL CHECK FOR SOLUTION =====

            # check for an all-ones row (solution)
            all_ones = np.all(button_matrix == 1, axis=1)

            # check if an all-ones row is present
            if np.any(all_ones):

                # update total and continue to next machine
                button_press_total += joltage_vector[np.where(all_ones)[0][0]]
                continue

            # ===== INITIAL SUMMATION =====

            # check if sum of rows in initial system yields solution

            # concatenation of all possible addition matrices and vectors
            updated_matrix, updated_vector = self.n_addition_matrix(
                button_matrix, joltage_vector)

            # check for an all-ones row
            all_ones = np.all(updated_matrix == 1, axis=1)

            # check if an all-ones row is present
            if np.any(all_ones):

                # update solution total and continue to next machine
                button_press_total += updated_vector[np.where(all_ones)[0][0]]
                continue

            # ===== MATRIX OPERATIONS & UPDATING SOLUTION VECTOR =====

            # OPERATION 1: update solution vector with original linear system

            updated_press_count = self.update_solution_vector(
                button_matrix, joltage_vector, updated_press_count)

            # check for solution
            if not np.any(updated_press_count['solution'] == -1):

                # update solution total and continue to next machine
                button_press_total += updated_press_count['solution'].sum()
                continue

            # OPERATION 2: update solution vector with reduced input

            updated_matrix, updated_vector = self.unique_solutions(
                button_matrix, joltage_vector)

            if np.any(updated_matrix):
                updated_press_count = self.update_solution_vector(
                    updated_matrix, updated_vector, updated_press_count)

            # check for solution
            if not np.any(updated_press_count['solution'] == -1):

                # update solution total and continue to next machine
                button_press_total += updated_press_count['solution'].sum()
                continue

            # OPERATION 3: update solution vector with difference matrix

            updated_press_count = self.update_solution_vector(
                *self.difference_matrix(button_matrix, joltage_vector),
                updated_press_count)

            # check for solution
            if not np.any(updated_press_count['solution'] == -1):

                # update solution total and continue to next machine
                button_press_total += updated_press_count['solution'].sum()
                continue

            # ===== LINEAR PROGRAMMING =====

            # minimisation of objective function (in this case, the solution \
            # vector variables) with equality constraints (the linear system) \
            # and output constraints (integer, or zero results)

            updated_press_count['solution'] = linprog(
                # coefficients of objective function, column vector of ones for
                # solution
                c=np.ones(press_count_vector.shape[0]),
                # constraint matrix / vector of coefficients of initial system
                A_eq=button_matrix,
                b_eq=joltage_vector,
                # bounds of each solution, generated by updated solution ranges
                bounds=[
                    (_['floor'], _['ceiling']) for _ in updated_press_count],
                method="highs",
                integrality=3  # integer or zero
                ).x[:press_count_vector.shape[0]]

            # check for solution, ensuring it solves the linear system
            if (not np.any(updated_press_count['solution'] == -1)) & \
                (np.allclose(
                    button_matrix @ updated_press_count['solution'],
                    joltage_vector)):

                # update solution total
                button_press_total += updated_press_count['solution'].sum()

            else:
                # otherwise print out unsolved machine details
                print(f"\nMachine # {idx} not solved:")
                print(updated_press_count['solution'])

            continue

        return button_press_total

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> float:
        """Solve of Part 1.

        Return minimum number of button presses for each indicator light \
        diagram.
        """
        # Calculate minimum button presses
        button_presses = self.configure_indicator_lights(
            indicator_lights=self.indicator_lights,
            buttons=self.buttons
            )

        # Derive sum of all presses
        self.solutions["Part One"] = sum(button_presses[0])

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return minimum number of button presses for each joltage level \
        counter.
        """
        # Calculate minimum button presses
        button_presses = self.configure_joltage_levels(
            joltage_levels=self.joltage_levels,
            buttons=self.buttons)

        # Max rectangle area
        self.solutions["Part Two"] = button_presses

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver10('Input10.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
