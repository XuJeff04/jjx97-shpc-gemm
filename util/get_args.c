#include "assignment3.h"


int get_args( int argc, char **argv, int *nrepeats, int *first, 
                                     int *last, int *inc )
{

     // For optional inputs                                                      
    char* p;                                                                    
    long int arg;                                                               
                                                    

    // Default values                            
    *nrepeats = 3;                                                               
                                                                                
    *first = 100;                                                                
    *last  = 500;                                                                
    *inc   = 50;                        
                                                                                
    if ( argc == 1 )                                                            
    {                                                                           
        printf("%% Using default values\n");                                    
    }                                                                           
                                                                                
    if ( argc >= 2 )                                                            
    {                                                                           
        arg = strtol(argv[1], &p, 10);                                          
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *nrepeats = (int ) arg;                                                  
    }                                                                           
    if ( argc >= 3 )                                                            
    {
        arg = strtol(argv[2], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *first = ( int ) arg;    
    }
    if ( argc >= 4 )                                                            
    {
        arg = strtol(argv[3], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *last = ( int ) arg;    
    }
    if ( argc >= 5 )                                                            
    {
        arg = strtol(argv[4], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *inc = ( int ) arg;    
    }
        
    printf("%% nrepeats = %d\n", *nrepeats);                                 
    printf("%% first = %d\n", *first);                                       
    printf("%% last = %d\n", *last);                                         
    printf("%% inc = %d\n", *inc);    
    return 0;




}
