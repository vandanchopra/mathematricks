1. [COMPLETED] Improve Strategy Performance 

1. lets create a new branch 
2. crypto_momentum_last.py is sending orders even when funds_available is not enough. This needs to be fixed - the issue appears to be in the position size calculation section where although funds_available is checked in calculate_position_details(), self.funds_available is incorrectly decremented for each signal's position_value but never reset, causing negative available funds (see line 361 and line 89). 

3. The equity curve being generated in systems/performance_reporter.py needs to be fixed. The issue appears to be in the generate_equity_curve() method where:
   - The calculation of position value in the loop may be incorrect since it's not properly tracking open vs closed positions
   - The timestamps comparison needs to handle timezone-aware and naive datetimes consistently
   - Missing proper initialization of portfolio values between timestamps