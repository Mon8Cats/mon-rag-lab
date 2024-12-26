# Error Handling

- respond to runtime errors
- indicate an error occurred
- exception hierarchies
- rethrow and wrap exceptions
- exception throwing code

## Why Handle Errors?

- Not crash program
- Chance to fix/retry
- Meaningful message & graceful exit
- Opportunity to log error
  
## Error Handling Using Error Codes

- need to know all the return values that represent errors
- need to know all the return values that represent success
- errors do not bubble up the call stack
- catch some errors at a higher level
- catch some errors in a single place
- how to deal with system errors
- how to return error form a constructor?
- not good

## Why Exceptions?

- don't need to know all error /success codes
- more readable, less clutter
- can bubble up
- catch exceptions in one place
- excepts are objects from System.Exception


## Understanding Exception

- exception bubbling
- try catch finally
- stack trace
- try{} 
  - catch(SomeException ex){} # most specific
  - catch(OtherException ex){}
  - catch(Exception ex){} # least specific or
  - catch {} # regardless of exception type
  - finally {} # always executed when control leaves the try block
- Unhandled error bubble up to OS level
  - Windows logs/Application/Error, application error, .net runtime, windows rrror reporting
- Stack Trace 
  - error - view details -> QuickWatch window
    - Expression: $exception object details
      - StackTrace property
- Custom Exception
  - var ex = new ArgumentOutOfRangeException("key", "message");
  - var ex = new ArgumentOutOfRangeException(nameof(operator), "message");
  - throw ex;
  - or throw new ArgumentOfRangeException(nameof(operator), "message");
- Visual Studio Debugger
- Windows event viewer
- Stack trace
- Threw exception
- catching exceptions
- exception handling good practices
    - do not add a catch block that does nothing or just rethrow
    - catch block should add some value?
    - may just be to log the error
    - usually bad practice to ignore (swallow/trap) exceptions
    - do not use exceptions for normal program flow logic
    - input validation: I expect input to be invalid sometimes
    - Parse => TryParse() // better?
- Design code to avoid exceptions 
  - check connection closed before close it?
  - consider returning null (or null object pattern) for extremely common errors
  
### System and Application Exceptions

- System: OutOFMemory, StackOverflow
- Third party: JsonSerialization
- My code: 
- the actual type of the exception class represents the kind of error that occurred.
- Hierarchy:
  - Exception -> SystemException, ArithmeticException, ApplicationException, customExecption
  - SystemException 
    - OutOfMemoryException, StackOverflowException, ArgumentException
      - ArgumentException -> ArgumentNullException, ArgumentOutOfRangeException,
  - ArithmeticException -> DivideByZeroException, OverflowException
- System.Exception properties
  - Message, StackTrace, Data, InnerException, Source
  - , HResult, HelpLink, TargetSite,  
- InnerException:
  - capture the preceding exception in new exception.
- Constructors
  - Excepetion(),
  - Exception(message)
  - Exception(message, innerException)
- System.ApplicationException for no-CLR exceptions

## Commonly Encountered Exceptions

- Exception & SystemException 
  - Do not throw?
  - Do not catch except in the top-level handlers?
  - Do not catch in framework code unless rethrowing
- InvalidOperationalException
- ArgumentException, ArgumentNullException, ArgumentOutOfRangeException
- NullReferenceException & IndexOutOfRangeException
- StackOverflowException
- OutOfMemoryException

## Catching, Throwing, Rethrowing


